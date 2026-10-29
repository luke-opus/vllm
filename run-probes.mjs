import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const dir=path.resolve('results/erf');
const zig=process.env.ZIG || 'zig';
const qemu=process.env.QEMU || 'qemu-system-riscv64';
const env={...process.env,ZIG_GLOBAL_CACHE_DIR:path.join(dir,'zig-cache'),ZIG_LOCAL_CACHE_DIR:path.join(dir,'zig-local-cache')};
const results=[];
function command(bin,args,limit=90000){const r=spawnSync(bin,args,{encoding:'utf8',env,timeout:limit});if(r.error)throw r.error;return r;}
for(const vlen of [128,256]) {
  const flags=['c++','-target','riscv64-linux-musl',`-mcpu=generic_rv64+m+a+f+d+c+v+zicsr+zifencei+zvfh+zvl${vlen}b`,`-mrvv-vector-bits=${vlen}`,'-mcmodel=medany','-std=c++17','-O2','-ffreestanding','-fno-exceptions','-fno-rtti','-fno-stack-protector','-fno-builtin','-ffp-contract=off','-nostdlib','-static',`-Wl,-T,${path.join(dir,'virt.ld')}`];
  for(const variant of ['baseline','fixed']) {
    const elf=path.join(dir,`${variant}-vlen${vlen}.elf`);
    const args=[...flags,path.join(dir,'start.S'),path.join(dir,variant+'.cpp'),'-o',elf];
    let r=command(zig,args,240000);
    fs.writeFileSync(path.join(dir,`${variant}-vlen${vlen}-build.log`),r.stdout+r.stderr);
    if(r.status)throw new Error(`${variant}/${vlen} build failed: ${r.stderr}`);
    for(const agnostic of [false,true]) {
      const cpu=`rv64,v=true,zvfh=true,vlen=${vlen},elen=64,rvv_ta_all_1s=${agnostic},rvv_ma_all_1s=${agnostic}`;
      const qargs=['-machine','virt','-cpu',cpu,'-m','128M','-nographic','-bios','none','-kernel',elf];
      r=command(qemu,qargs,30000);
      const output=r.stdout+r.stderr;
      fs.writeFileSync(path.join(dir,`${variant}-vlen${vlen}-agnostic${Number(agnostic)}.log`),output);
      const match=output.match(/checks=(\d+) failures=(\d+) nan_failures=(\d+) sign_failures=(\d+)/);
      if(!match)throw new Error(`Missing results ${variant}/${vlen}: ${output}`);
      const record={variant,vlen,agnostic_ones:agnostic,returncode:r.status,checks:+match[1],failures:+match[2],nan_failures:+match[3],sign_failures:+match[4],build_command:[zig,...args],run_command:[qemu,...qargs]};
      results.push(record);fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(results,null,2)+'\n');
      console.log(JSON.stringify({...record,build_command:undefined,run_command:undefined}));
    }
  }
}

if (!results.some(r => r.variant === 'baseline' && r.agnostic_ones && r.nan_failures > 0) || results.some(r => r.variant === 'fixed' && (r.returncode !== 0 || r.failures || r.sign_failures))) throw new Error('Unexpected comparison result');
