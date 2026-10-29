import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const dir=path.resolve('results/erf');
const zig=process.env.ZIG || 'zig';
const qemu=process.env.QEMU || 'qemu-system-riscv64';
const env={...process.env,ZIG_GLOBAL_CACHE_DIR:path.join(dir,'zig-cache'),ZIG_LOCAL_CACHE_DIR:path.join(dir,'zig-local-cache')};
const harness=`template<class V>void special(){float in[16],out[16];float values[]={0.0f,-0.0f,__builtin_inff(),-__builtin_inff(),__builtin_nanf(""),-__builtin_nanf(""),1.0f,-1.0f};float refs[]={0.0f,-0.0f,1.0f,-1.0f,__builtin_nanf(""),__builtin_nanf(""),0.84270079295f,-0.84270079295f};for(int i=0;i<V::VEC_ELEM_NUM;++i)in[i]=values[i%8];V(in).er().save(out);for(int i=0;i<V::VEC_ELEM_NUM;++i){++checks;float x=out[i],ref=refs[i%8];if(nan(ref)){if(!nan(x))++failures;}else if(nan(x)||absolute(x-ref)>1e-6f){++failures;if(nan(x))++nan_failures;}if(in[i]==0.0f&&__builtin_signbit(x)!=__builtin_signbit(in[i]))++sign_failures;}}\nextern "C" int test_main(){special<FP32Vec8>();special<FP32Vec16>();text("checks=");number(checks);text(" failures=");number(failures);text(" nan_failures=");number(nan_failures);text(" sign_failures=");number(sign_failures);text("\\n");return failures||sign_failures?1:0;}\n`;
const results=[];
for(const variant of ['baseline','fixed']) {
  const original=fs.readFileSync(path.join(dir,variant+'.cpp'),'utf8');
  fs.writeFileSync(path.join(dir,'special-'+variant+'.cpp'),original.slice(0,original.indexOf('extern "C" int test_main()'))+harness);
  for(const vlen of [128,256]) {
    const elf=path.join(dir,`special-${variant}-vlen${vlen}.elf`);
    const args=['c++','-target','riscv64-linux-musl',`-mcpu=generic_rv64+m+a+f+d+c+v+zicsr+zifencei+zvfh+zvl${vlen}b`,`-mrvv-vector-bits=${vlen}`,'-mcmodel=medany','-std=c++17','-O2','-ffreestanding','-fno-exceptions','-fno-rtti','-fno-stack-protector','-fno-builtin','-ffp-contract=off','-nostdlib','-static',`-Wl,-T,${path.join(dir,'virt.ld')}`,path.join(dir,'start.S'),path.join(dir,`special-${variant}.cpp`),'-o',elf];
    let r=spawnSync(zig,args,{encoding:'utf8',env,timeout:60000});
    if(r.status)throw new Error(r.stderr);
    for(const agnostic of [false,true]) {
      const cpu=`rv64,v=true,zvfh=true,vlen=${vlen},elen=64,rvv_ta_all_1s=${agnostic},rvv_ma_all_1s=${agnostic}`;
      const qargs=['-machine','virt','-cpu',cpu,'-m','128M','-nographic','-bios','none','-kernel',elf];
      r=spawnSync(qemu,qargs,{encoding:'utf8',env,timeout:30000});
      const output=r.stdout+r.stderr;
      fs.writeFileSync(path.join(dir,`special-${variant}-vlen${vlen}-agnostic${Number(agnostic)}.log`),output);
      const match=output.match(/checks=(\d+) failures=(\d+) nan_failures=(\d+) sign_failures=(\d+)/);
      if(!match)throw new Error(output);
      const record={variant,vlen,agnostic_ones:agnostic,returncode:r.status,checks:+match[1],failures:+match[2],nan_failures:+match[3],sign_failures:+match[4]};
      results.push(record);console.log(JSON.stringify(record));
    }
  }
}
fs.writeFileSync(path.join(dir,'special-results.json'),JSON.stringify(results,null,2)+'\n');

if (results.some(r => r.variant === 'fixed' && (r.returncode !== 0 || r.failures || r.sign_failures))) throw new Error('Fixed special-value regression');
