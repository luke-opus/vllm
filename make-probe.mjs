import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import path from 'node:path';
const [repoArg, base, fixed] = process.argv.slice(2);
if (!repoArg || !base || !fixed) throw new Error('Usage: node make-probe.mjs REPO BASE_REF FIXED_REF');
const repo = path.resolve(repoArg);
const dir = path.resolve('results/erf');
fs.mkdirSync(dir, {recursive: true});
function read(ref, file) {
  const r = spawnSync('git', ['-C', repo, 'show', ref + ':' + file], {encoding:'utf8'});
  if (r.status !== 0) throw new Error(r.stderr);
  return r.stdout;
}
fs.writeFileSync(path.join(dir, 'cpu_types_riscv_defs.hpp'), read(fixed, 'csrc/cpu/cpu_types_riscv_defs.hpp'));
for (const file of ['start.S', 'virt.ld']) fs.copyFileSync(new URL(file, import.meta.url), path.join(dir,file));
for (const [variant, ref] of [['baseline',base],['fixed',fixed]]) {
const source = read(ref, 'csrc/cpu/cpu_types_riscv_impl.hpp');
function method(name, type) {
  const start = source.indexOf(`  ${type} ${name}() const {`);
  if (start < 0) throw new Error('Missing source method');
  const brace = source.indexOf('{', start);
  let depth = 1, end = brace + 1;
  for (; depth; ++end) {
    if (source[end] === '{') ++depth;
    if (source[end] === '}') --depth;
  }
  return source.slice(start,end);
}
let classes = '';
for (const width of [8,16]) {
  const type=`FP32Vec${width}`, lmul=`LMUL_${width*32}`, reg=`fixed_fp32x${width}_t`;
  classes += `struct ${type} {\n static constexpr int VEC_ELEM_NUM=${width};\n ${reg} reg;\n explicit ${type}(${reg} r):reg(r){}\n explicit ${type}(const float* p):reg(RVVI(__riscv_vle32_v_f32, ${lmul})(p,VEC_ELEM_NUM)){}\n void save(float* p)const {RVVI(__riscv_vse32_v_f32, ${lmul})(p,reg,VEC_ELEM_NUM);}\n${method('exp',type)}\n${method('er',type)}\n};\n`;
}
const values=[0,-0,0.0001,-0.0001,0.125,-0.125,0.25,-0.25,0.5,-0.5,0.75,-0.75,1,-1,1.5,-1.5,2,-2,3,-3,4,-4,6,-6,9,-9];
for(let i=0;i<230;i++) values.push(((i*73)%257-128)/16);
while(values.length%16) values.push(1);
const host=`#include <cmath>\n#include <cstdio>\nint main(){double values[]={${values.map(v=>Object.is(v,-0)?'-0.0':String(v)).join(',')}};\nputs("static const float inputs[]={");for(double x:values)printf("%.9gf,\\n",x);puts("};");puts("static const float expected[]={");for(double x:values)printf("%.9gf,\\n",std::erf(x));puts("};");}\n`;
fs.writeFileSync(path.join(dir,'reference.cpp'),host);
let r=spawnSync(process.env.CXX || 'c++',[path.join(dir,'reference.cpp'),'-std=c++17','-O2','-o',path.join(dir,'reference')],{encoding:'utf8'});
if(r.status) throw new Error(r.stderr);
r=spawnSync(path.join(dir,'reference'),[],{encoding:'utf8'});
if(r.status) throw new Error(r.stderr);
// C++ requires a decimal point in literals such as 0.0f and 1.0f.
const data=r.stdout.replace(/(?<![\w.])(-?\d+)f/g,'$1.0f');
const harness=`\nstatic void text(const char* s){volatile char* p=(volatile char*)0x10000000;while(*s)*p=*s++;}\nstatic void number(unsigned n){char b[16];int len=0;do{b[len++]='0'+n%10;n/=10;}while(n);while(len){char s[2]={b[--len],0};text(s);}}\nstatic float absolute(float x){return x<0?-x:x;}\nstatic bool nan(float x){return __builtin_isnan(x);}\nstatic unsigned checks=0, failures=0, nan_failures=0, sign_failures=0;\ntemplate<class V>void check(){float out[16];for(unsigned i=0;i<sizeof(inputs)/sizeof(float);i+=V::VEC_ELEM_NUM){V in(inputs+i);in.er().save(out);for(int j=0;j<V::VEC_ELEM_NUM;++j){++checks;const float x=out[j],ref=expected[i+j];if(nan(x)||absolute(x-ref)>1e-6f+1e-5f*absolute(ref)){++failures;if(nan(x))++nan_failures;}if(inputs[i+j]==0.0f&&__builtin_signbit(x)!=__builtin_signbit(inputs[i+j]))++sign_failures;}}}\nextern "C" int test_main(){check<FP32Vec8>();check<FP32Vec16>();text("checks=");number(checks);text(" failures=");number(failures);text(" nan_failures=");number(nan_failures);text(" sign_failures=");number(sign_failures);text("\\n");return failures||sign_failures?1:0;}\n`;
fs.writeFileSync(path.join(dir,variant+'.cpp'),`#include "cpu_types_riscv_defs.hpp"\n${classes}\n${data}\n${harness}`);
console.log(`Extracted ${ref}: ${values.length} host-libm cases per vector width`);
}
