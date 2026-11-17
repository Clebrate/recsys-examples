#!/usr/bin/env python3
"""Opt-in actual NVEmbedding test; only run on the authorized GPU allocation.
python native_nvemb_general.py --runtime-root /installed/nve --output result.json --prepared
The caller supplies PYTHONPATH/LD_LIBRARY_PATH for that same runtime.
"""
import argparse,json,os,time,traceback
from pathlib import Path
M=1<<20

def decode(x): return json.loads(x) if isinstance(x,str) else x

def run(args,report):
 import numpy as np
 import torch
 import pynve.nve as nve
 from pynve.torch import nve_layers,nve_ps
 module=Path(nve.__file__).resolve()
 assert module.is_relative_to(args.runtime_root.resolve()),str(module)
 torch.cuda.set_device(args.device)
 device=torch.device('cuda',args.device)
 report.update(module=str(module),device=torch.cuda.get_device_name(args.device),stages=[],prepared_resize=args.prepared)
 row=64;ways=8
 def make(payload,general=True,capacity=65536,maximum=4*M,backing_keys=None):
  os.environ['NVE_NVEMB_VMM_MAX_PAYLOAD_BYTES']=str(maximum if general else 0)
  os.environ['NVE_NVEMB_VMM_CHUNK_BYTES']=str(2*M)
  os.environ['NVE_NVEMB_VMM_GENERAL_REHASH']=str(int(general))
  os.environ['NVE_NVEMB_POLICY_OBSERVATION']='1'
  os.environ['NVE_NVEMB_POLICY_EVENT_CAPACITY']=str(capacity)
  ps=nve_ps.NVEParameterServer(num_embeddings=0,embedding_size=row//4,data_type=torch.float32,initializer=None,initial_size=1024)
  if backing_keys is not None:
   keys=torch.as_tensor(backing_keys,dtype=torch.int64,device=device).contiguous()
   ps.insert(keys.cpu(),value(keys).cpu())
  sets=payload//(ways*row)
  emb=nve_layers.NVEmbedding(num_embeddings=1<<24,embedding_size=row//4,data_type=torch.float32,
   layer_type=nve_layers.LayerType.Hierarchical,gpu_cache_size=sets*(ways*row+104),host_cache_size=0,
   storage=ps,optimize_for_training=False,device=device,config={'max_modify_size':65536})
  torch.cuda.synchronize()
  state=decode(emb.emb_layer.cache_resize_state())
  assert state['vmm']==general and state['payload_bytes']==payload
  if general:
   assert state['per_row_ages'] and state['rehash_mode']=='native_modulo_general_row_ages_cpu_staging'
  return emb
 def value(keys): return keys[:,None].to(torch.float32)+torch.arange(row//4,device=device,dtype=torch.float32)[None,:]/32+1
 def insert(emb,keys):
  k=torch.as_tensor(keys,dtype=torch.int64,device=device).contiguous();v=value(k)
  torch.cuda.synchronize();emb.insert(k,v,table_id=0);torch.cuda.synchronize()
 def snap(emb):
  b=emb.emb_layer;s=decode(b.cache_resize_state());n=s['num_sets'];t=np.empty(n*8,np.int64);c=np.empty(n*8,np.uint32)
  general=s.get('per_row_ages',False);a=np.empty(n*8 if general else n,np.int64)
  f=b.cache_resize_row_metadata if general else b.cache_resize_metadata
  s=decode(f(n,t.ctypes.data,c.ctypes.data,a.ctypes.data));a=a if general else np.repeat(a,8)
  valid=np.flatnonzero(t>=0);keys=t[valid]*n+valid//8;order=np.argsort(keys)
  return {'state':s,'keys':keys[order],'bits':c[valid][order],'ages':a[valid][order]}
 def same(a,b):
  for field in ('keys','bits','ages'): assert np.array_equal(a[field],b[field]),field
  assert a['state']['global_ts']==b['state']['global_ts']
 def verify_values(emb,keys):
  for first in range(0,len(keys),8192):
   k=torch.as_tensor(keys[first:first+8192],device=device,dtype=torch.int64).contiguous()
   expected=value(k);actual=emb(k);torch.cuda.synchronize();assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
 def observe(emb,reset=True):
  o=decode(emb.emb_layer.cache_policy_observation(reset));assert o['enabled'] and o['resident_rows']>=0
  assert o['ps_find_enabled'];assert o['resident_rows']==len(snap(emb)['keys']), 'Resident reduction differs from raw native tags';return o
 def resize(emb,target):
  before=snap(emb);b=emb.emb_layer;n=target//(8*row)
  score=before['bits'].view(np.float32).astype(np.float64)*np.power(float(before['state']['decay_rate']),before['state']['global_ts']-before['ages'])
  dest=before['keys']%n;order=np.lexsort((before['keys'],-score,dest));d=dest[order]
  start=np.maximum.accumulate(np.where(np.r_[True,d[1:]!=d[:-1]],np.arange(len(d)),0))
  keep=order[np.arange(len(order))-start<8];keep=keep[np.argsort(before['keys'][keep])]
  plan=decode((b.prepare_cache_resize if args.prepared else b.plan_cache_resize)(target,40<<30))
  assert plan['status']==('prepared' if args.prepared else 'planned'),plan
  assert plan['host_staging_peak_bound_bytes']<=40<<30
  applied=decode(b.commit_cache_resize(plan['token'],32<<30,40<<30) if args.prepared else
                 b.apply_cache_resize(target,plan['generation'],32<<30,40<<30));assert applied['status']=='applied',applied
  if args.prepared:
   assert applied['plan_reused'] and applied['token_consumed'] and applied['native_preflight_ms']==0
   assert not decode(b.cache_resize_state())['prepared']
   stale=decode(b.commit_cache_resize(plan['token'],32<<30,40<<30))
   assert stale['status']=='rejected' and not stale['token_consumed']
  after=snap(emb)
  for field in ('keys','bits','ages'):assert np.array_equal(after[field],before[field][keep]),field
  assert after['state']['global_ts']==before['state']['global_ts']
  assert after['state']['value_base_address']==before['state']['value_base_address']
  assert after['state']['device_pool_bytes']==after['state']['allocated_value_bytes']+after['state']['fixed_metadata_bytes']+after['state']['observation_device_bytes']
  assert applied['evicted_rows']==len(before['keys'])-len(after['keys'])
  assert applied['contexts_rebuilt_before_return']
  verify_values(emb,after['keys'])
  return before,after,applied
 legacy=make(2*M,False);general=make(2*M);n=2*M//(8*row)
 for first in range(0,n,1024):
  keys=(np.arange(first,min(first+1024,n),dtype=np.int64)[:,None]+np.arange(8,dtype=np.int64)*n).ravel()
  weighted=np.repeat(keys,keys//n+1)
  insert(legacy,weighted);insert(general,weighted)
 frequent=np.repeat(np.arange(16,dtype=np.int64),8);insert(legacy,frequent);insert(general,frequent)
 a=snap(legacy);b=snap(general);same(a,b);verify_values(legacy,a['keys']);verify_values(general,b['keys'])
 assert observe(legacy)['admission_eviction_count']==0;assert observe(general)['admission_eviction_count']==0
 report['stages'].append('original_vs_general_multibatch_LFU_bits_ages_values_equal')
 if args.prepared:
  binding=general.emb_layer;caps=decode(binding.cache_resize_capabilities())
  assert caps['prepared_resize'] and caps['prepared_resize_version']==1
  before=snap(general)
  plan=decode(binding.prepare_cache_resize(3*M,40<<30));assert plan['status']=='prepared',plan
  token=plan['token'];assert isinstance(token,str) and token
  assert plan['prepared_host_bytes']>0
  assert decode(binding.cache_resize_state())['prepared_token']==token
  duplicate=decode(binding.prepare_cache_resize(M,40<<30));assert duplicate['status']=='rejected'
  invalid=token+'invalid'
  for operation in (lambda:binding.cancel_cache_resize(invalid),
                    lambda:binding.commit_cache_resize(invalid,32<<30,40<<30)):
   rejected=decode(operation());assert rejected['status']=='rejected' and not rejected['token_consumed']
   assert decode(binding.cache_resize_state())['prepared_token']==token
  k=torch.as_tensor(before['keys'][:1],device=device,dtype=torch.int64).contiguous()
  # These real public calls must fail before enqueueing while the saved plan is live.
  for operation in (lambda:general(k),lambda:general.insert(k,value(k),table_id=0)):
   try:operation()
   except RuntimeError as error:assert 'resize' in str(error).lower(),str(error)
   else:raise AssertionError('Prepared cache accepted a data operation')
  torch.cuda.synchronize()
  cancelled=decode(binding.cancel_cache_resize(token));assert cancelled['status']=='cancelled'
  assert not decode(binding.cache_resize_state())['prepared'];same(before,snap(general))
  stale=decode(binding.cancel_cache_resize(token));assert stale['status']=='rejected' and not stale['token_consumed']
  verify_values(general,before['keys'])
  for pool_limit,host_limit in ((1,40<<30),(32<<30,1)):
   before=snap(general);plan=decode(binding.prepare_cache_resize(3*M,40<<30))
   assert plan['status']=='prepared'
   rejected=decode(binding.commit_cache_resize(plan['token'],pool_limit,host_limit))
   assert rejected['status']=='rejected' and rejected['token_consumed']
   assert rejected['contexts_rebuilt_before_return'] and not decode(binding.cache_resize_state())['prepared']
   same(before,snap(general));verify_values(general,before['keys'])
  report['stages'].append('prepared_cancel_stale_token_blocked_data_calls_and_budget_rejection_restore_contexts')
  # Native admission accumulates 65,536 input rows. Preserve a half-full
  # collection across the transaction, then finish it with disjoint keys.
  pending_keys=np.arange(2048,dtype=np.int64)
  controls=[]
  for transition in ('none','cancel','commit'):
   pending=make(2*M,backing_keys=pending_keys);binding=pending.emb_layer
   first=torch.as_tensor(np.tile(pending_keys[:1024],32),device=device,dtype=torch.int64)
   actual=pending(first);torch.cuda.synchronize();assert torch.equal(actual,value(first))
   assert observe(pending)['resident_rows']==0, 'Fixture already admitted its first half'
   if transition!='none':
    plan=decode(binding.prepare_cache_resize(3*M,40<<30));assert plan['status']=='prepared'
    if transition=='cancel':assert decode(binding.cancel_cache_resize(plan['token']))['status']=='cancelled'
    else:assert decode(binding.commit_cache_resize(plan['token'],32<<30,40<<30))['status']=='applied'
   second=torch.as_tensor(np.tile(pending_keys[1024:],32),device=device,dtype=torch.int64)
   actual=pending(second);torch.cuda.synchronize();assert torch.equal(actual,value(second))
   observation=observe(pending);state=snap(pending)
   assert observation['resident_rows']==len(pending_keys), 'Pending admission rows were lost'
   assert np.array_equal(state['keys'],pending_keys)
   controls.append(state);verify_values(pending,pending_keys)
  same(controls[0],controls[1]);same(controls[0],controls[2])
  report['stages'].append('pending_half_threshold_auto_insert_survives_cancel_and_commit_exact_values_bits_ages')
 receipts=[]
 for target in (3*M,M):
  before,after,receipt=resize(general,target);receipts.append(receipt)
  o=observe(general);victims=np.setdiff1d(before['keys'],after['keys'])
  assert o['complete'] and not o['overflow'] and o['capacity_eviction_count']==len(victims)
  assert np.array_equal(np.sort(np.asarray(o['capacity_eviction_keys'],dtype=np.int64)),victims)
  assert o['admission_eviction_count']==0
 report['resize_receipts']=receipts;report['stages'].append('arbitrary_mixed_age_growth_and_shrink_exact_survivors_and_capacity_events')
 before=snap(general);new=np.arange(16,dtype=np.int64)+(1<<20)
 next_clock=before['state']['global_ts']+1;sets=before['state']['num_sets']
 effective=before['bits'].view(np.float32).astype(np.float64)*np.power(float(before['state']['decay_rate']),next_clock-before['ages'])
 expected_victims=[];victim_relative_gaps=[]
 for bucket in new%sets:
  candidates=np.flatnonzero(before['keys']%sets==bucket);scores=effective[candidates]
  assert len(np.unique(before['ages'][candidates]))>1, 'Postresize victim fixture must contain mixed ages'
  ranked=np.sort(scores)
  relative_gap=float(ranked[1]-ranked[0])/max(float(abs(ranked[0])),float(abs(ranked[1])),float(np.finfo(np.float32).tiny))
  assert relative_gap>1e-3, 'Victim fixture must have a separated relative minimum'
  victim_relative_gaps.append(relative_gap)
  expected_victims.append(before['keys'][candidates[np.argmin(scores)]])
 inserted_keys=np.repeat(new,50)
 expected_new_counter=np.float32(50)*np.float32(1.0/len(inserted_keys))
 insert(general,inserted_keys);after=snap(general);o=observe(general)
 victims=np.setdiff1d(before['keys'],after['keys'])
 assert np.array_equal(victims,np.sort(np.asarray(expected_victims,np.int64))), 'Wrong LFU victims after mixed-age migration'
 assert len(victims)==16 and o['admission_eviction_count']==16 and o['capacity_eviction_count']==0
 assert after['state']['global_ts']==next_clock
 before_index={int(key):i for i,key in enumerate(before['keys'])};new_set=set(map(int,new));affected=set(map(int,new%sets))
 for i,key in enumerate(after['keys']):
  if int(key) in new_set:
   assert after['ages'][i]==next_clock and after['bits'][i:i+1].view(np.float32)[0]==expected_new_counter
  else:
   source=before_index[int(key)]
   if int(key%sets) in affected:
    assert after['ages'][i]==next_clock
    actual=float(after['bits'][i:i+1].view(np.float32)[0])
    assert np.isclose(actual,effective[source],rtol=5e-6,atol=0), 'Incorrect postinsert lazy decay'
   else:
    assert after['ages'][i]==before['ages'][source] and after['bits'][i]==before['bits'][source]
 report['post_insert_oracle']={'mixed_age_buckets':len(new),'victim_minima_strictly_separated':True,'victim_keys_exact':True,'row_ages_exact':True,'untouched_counter_bits_exact':True,'touched_counter_numeric_rtol':5e-6,'minimum_victim_relative_gap':min(victim_relative_gaps),'new_counter_expected':float(expected_new_counter),'new_counter_rule':'float32(count) * float32(1 / batch_input_rows)'}
 assert np.array_equal(np.sort(np.asarray(o['admission_eviction_keys'],dtype=np.int64)),victims)
 assert np.all(np.isin(new,after['keys']));verify_values(general,after['keys'])
 report['stages'].append('continue_original_insert_after_rehash_and_actual_admission_evictions')
 before=snap(general)
 rejected=decode(general.emb_layer.plan_cache_resize(3*M,1));assert rejected['status']=='rejected';same(before,snap(general))
 stale=decode(general.emb_layer.apply_cache_resize(3*M,before['state']['generation']+99,32<<30,40<<30));assert stale['status']=='rejected';same(before,snap(general));verify_values(general,before['keys'])
 report['stages'].append('host_budget_and_stale_generation_reject_without_mutation_contexts_remain_usable')
 observe(general)
 missing=np.asarray([(1<<23)+i for i in range(5)],np.int64)
 general(torch.as_tensor(missing,device=device));torch.cuda.synchronize();o=observe(general)
 assert o['ps_find_complete'] and o['ps_find_count']==len(missing) and o['ps_find_calls']>0 and o['ps_find_work_ms']>=0
 assert np.array_equal(np.sort(np.asarray(o['ps_find_keys'],dtype=np.int64)),missing)
 report['ps_find_evidence']={k:v for k,v in o.items() if k.startswith('ps_find')};report['stages'].append('actual_native_PS_find_ids_and_work_observed')
 small=make(128*8*row,True,capacity=2,maximum=2*M);insert(small,np.arange(1024,dtype=np.int64));observe(small)
 insert(small,np.repeat((1<<20)+np.arange(16,dtype=np.int64),100));overflow=observe(small)
 assert overflow['admission_eviction_count']==16 and overflow['overflow'] and not overflow['complete']
 assert len(overflow['admission_eviction_keys'])==2 and overflow['dropped_event_count']==14
 empty=observe(small);assert empty['complete'] and not empty['overflow'] and empty['admission_eviction_count']==0
 assert empty['observation_serial']==overflow['observation_serial']+1
 report['overflow_evidence']=overflow;report['stages'].append('bounded_event_overflow_unknown_and_reset_serial')
 report['status']='passed'

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--prepared',action='store_true');a=p.parse_args()
 report={'status':'running','started':time.time()}
 try:run(a,report)
 except BaseException:report.update(status='failed',traceback=traceback.format_exc())
 report['elapsed_s']=time.time()-report['started'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'stages':report.get('stages',[]),'output':str(a.output),'elapsed_s':report['elapsed_s']}));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
