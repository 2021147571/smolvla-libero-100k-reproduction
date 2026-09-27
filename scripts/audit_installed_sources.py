import importlib.metadata as m,hashlib,base64,json
out={}
for name in ['lerobot','hf_libero','robosuite']:
 try:d=m.distribution(name)
 except m.PackageNotFoundError:continue
 changed=[]; count=0
 for f in d.files or []:
  if not str(f).endswith('.py') or not f.hash:continue
  p=d.locate_file(f)
  if not p.exists(): changed.append({'file':str(f),'missing':True});continue
  count+=1
  h=base64.urlsafe_b64encode(hashlib.new(f.hash.mode,p.read_bytes()).digest()).decode().rstrip('=')
  if h!=f.hash.value:changed.append({'file':str(f),'changed':True})
 out[name]={'version':d.version,'checked_python_files':count,'changes':changed}
print(json.dumps(out,indent=2))
