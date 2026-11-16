#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exact 3 GiB native HSTU KV resize, with prefilled synthetic page bytes.

No model forward or KV generation runs. Public native page allocation registers
full resident histories before timing; fill and byte checks are outside timing.
All GPU work is serial on one exclusively owned allocation.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

GIB=1024**3
MIB=1024**2
PAGE_BYTES=65536

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def run(args):
    sys.path.insert(0,str(args.repo_root))
    import torch
    from examples.hstu.inference.cache_ratio.memory_manager.hstu import HstuVmmBackend
    torch.cuda.init()
    context_probe=torch.empty(1,device="cuda")
    torch.cuda.synchronize()
    del context_probe
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    records=[]
    for repeat in range(args.repeats):
        cases=[(m,d) for m in (2,16,64) for d in ('grow','shrink')]
        if repeat%2: cases.reverse()
        for mib,direction in cases:
            name=f'r{repeat}-{mib}MiB-{direction}'
            path=root/(name+'.json')
            before_pages=(GIB if direction=='grow' else 2*GIB)//PAGE_BYTES
            after_pages=(2*GIB if direction=='grow' else GIB)//PAGE_BYTES
            record=dict(case=name,repeat=repeat,direction=direction,chunk_mib=mib,
                        backend='native_hstu_gpu_only_vmm',status='starting',
                        fixture='synthetic_fully_resident_native_histories_no_model_forward',
                        transferred_payload_bytes=3*GIB)
            write(path,record)
            backend=None
            try:
                config=SimpleNamespace(host_kvstorage_backend='nop',host_capacity_per_layer=0,
                    num_buffer_pages=0,num_layers=3,num_heads=4,head_dim=128,page_size=32,
                    offload_chunksize=32,num_primary_cache_pages=before_pages,
                    max_batch_size=32,max_seq_len=1024,dtype=torch.float16,device=0)
                backend=HstuVmmBackend(config,max_pages=2*GIB//PAGE_BYTES,chunk_bytes=mib*MIB)
                users=before_pages//32
                for first in range(0,users,32):
                    ids=torch.arange(first,min(first+32,users),dtype=torch.int64)
                    lengths=torch.full((len(ids),),1024,dtype=torch.int64)
                    with backend.batch():
                        index,look=backend.lookup_kvcache(ids,lengths)
                        metadata=backend.allocate_kvcache(index,look)
                        del metadata,index,look
                # Fill every byte in the resident KV pool, outside resize timing.
                for layer in range(3):
                    backend.get_cache_tables(layer).fill_(layer+1)
                torch.cuda.synchronize()
                snapshot_before=backend.impl_._vmm_snapshot()
                before=backend.stats()
                assert len(snapshot_before)==users
                assert before['resident_payload_bytes']==before['payload_bytes']==3*before_pages*PAGE_BYTES
                pointers=tuple(t.data_ptr() for t in backend.get_cache_tables())
                record.update(before=before,resident_users_before=users,
                              resident_tokens_before=users*1024)
                native_pool_resize=backend._pool.resize
                pool_timings=[]
                def measured_pool(*a,**kw):
                    started=time.perf_counter_ns()
                    result=native_pool_resize(*a,**kw)
                    pool_timings.append((time.perf_counter_ns()-started)/1e6)
                    return result
                backend._pool.resize=measured_pool
                torch.cuda.synchronize()
                started=time.perf_counter_ns()
                result=backend.resize_pages(after_pages,at_batch_barrier=True)
                returned=time.perf_counter_ns()
                torch.cuda.synchronize()
                ready=time.perf_counter_ns()
                backend._pool.resize=native_pool_resize
                after=backend.stats()
                snapshot_after=backend.impl_._vmm_snapshot()
                assert after['payload_bytes']==3*after_pages*PAGE_BYTES
                assert abs(after['payload_bytes']-before['payload_bytes'])==3*GIB
                assert tuple(t.data_ptr() for t in backend.get_cache_tables())==pointers
                old={r[0]:r[1:] for r in snapshot_before}
                for row in snapshot_after:
                    assert old[row[0]]==row[1:], 'survivor metadata changed'
                retained=min(before_pages,after_pages)
                # Full retained-value verification, in bounded slices, no sampling.
                for layer in range(3):
                    for first in range(0,retained,512):
                        assert bool(torch.all(backend.get_cache_tables(layer)[first:first+min(512,retained-first)]==layer+1))
                    if direction=='grow':
                        for first in range(before_pages,after_pages,512):
                            assert bool(torch.all(backend.get_cache_tables(layer)[first:first+min(512,after_pages-first)]==0))
                expected_users=min(users,after_pages//32)
                assert len(snapshot_after)==expected_users
                assert len(result['evicted_users'])==users-expected_users
                # Existing user histories are still found through the public lookup.
                kept_ids=[r[0] for r in snapshot_after]
                for first in range(0,len(kept_ids),32):
                    ids=torch.tensor(kept_ids[first:first+32],dtype=torch.int64)
                    lengths=torch.full((len(ids),),1024,dtype=torch.int64)
                    with backend.batch():
                        index,look=backend.lookup_kvcache(ids,lengths)
                        assert bool(torch.all(look.cached_lengths==1024))
                        del index,look
                record.update(status='passed',after=after,result=result,
                    call_ms=(returned-started)/1e6,total_ms=(ready-started)/1e6,
                    vmm_pool_ms=sum(pool_timings),resident_users_after=expected_users,
                    evicted_users=users-expected_users,
                    retained_bytes_fully_checked=3*retained*PAGE_BYTES,
                    new_zero_bytes_fully_checked=3*max(0,after_pages-before_pages)*PAGE_BYTES,
                    stable_addresses=True)
            except BaseException as exc:
                record.update(status='failed',error=repr(exc))
                raise
            finally:
                if backend is not None:
                    try:
                        backend.close(at_batch_barrier=True)
                        record['cleanup']='closed'
                    except BaseException as exc:
                        record['cleanup_error']=repr(exc)
                        record['status']='failed'
                write(path,record)
            records.append(record)
            print(name,record['status'],record.get('total_ms'),flush=True)
    groups=[]
    for mib in (2,16,64):
        for direction in ('grow','shrink'):
            rows=[r for r in records if r['chunk_mib']==mib and r['direction']==direction]
            groups.append(dict(chunk_mib=mib,direction=direction,n=len(rows),
                total_ms_mean=statistics.mean(r['total_ms'] for r in rows),
                total_ms_min=min(r['total_ms'] for r in rows),
                total_ms_max=max(r['total_ms'] for r in rows),
                vmm_pool_ms_mean=statistics.mean(r['vmm_pool_ms'] for r in rows)))
    write(root/'summary.json',dict(status='passed',cases=records,groups=groups))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    run(parser.parse_args())
