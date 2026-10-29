import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const [repoArg, base, fixed] = process.argv.slice(2);
if (!repoArg || !base || !fixed) throw new Error('Usage: node run-bf16.mjs REPO BASE_REF FIXED_REF');
const repo = path.resolve(repoArg);
const dir = path.resolve('results/bf16');
fs.mkdirSync(dir, {recursive: true});
const results = [];
for (const [variant, ref] of [['baseline', base], ['fixed', fixed]]) {
  const include = path.join(dir, variant);
  fs.mkdirSync(include, {recursive: true});
  const header = spawnSync('git', ['-C', repo, 'show', ref + ':csrc/cpu/float_convert.hpp'], {encoding: 'utf8'});
  if (header.status !== 0) throw new Error(header.stderr);
  fs.writeFileSync(path.join(include, 'float_convert.hpp'), header.stdout);
  const binary = path.join(include, 'conversion');
  const args = ['-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-I' + include, new URL('convert_test.cpp', import.meta.url).pathname, '-o', binary];
  const compile = spawnSync(process.env.CXX || 'c++', args, {encoding: 'utf8'});
  if (compile.status !== 0) throw new Error(compile.stderr);
  const run = spawnSync(binary, [], {encoding: 'utf8', timeout: 60000});
  fs.writeFileSync(path.join(dir, variant + '.log'), run.stdout + run.stderr);
  console.log(variant + ':\n' + run.stdout);
  results.push({variant, ref, returncode: run.status, output: run.stdout});
}
fs.writeFileSync(path.join(dir, 'results.json'), JSON.stringify(results, null, 2) + '\n');
if (results[0].returncode !== 1 || results[1].returncode !== 0) throw new Error('Unexpected comparison result');
