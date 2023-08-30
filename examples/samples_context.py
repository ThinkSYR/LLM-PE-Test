# -*-coding:utf-8-*-
import json
import random

contexts_base = json.load(open('data/contexts.json'))
contexts_other = json.load(open('data/contexts.100.json'))

contexts_base = [c.replace('\n\n', '\n') for c in contexts_base]
contexts_other = [c.replace('\n\n', '\n') for c in contexts_other]

contexts1 = contexts_base
context1 = '\n\n'.join(contexts1)
with open(f"data/context.{len(context1)}.txt", 'w') as f:
    f.write(context1)

contexts2 = contexts_base + contexts_other[:5]
random.shuffle(contexts2)
context2 = '\n\n'.join(contexts2)
with open(f"data/context.{len(context2)}.txt", 'w') as f:
    f.write(context2)

contexts3 = contexts_base + contexts_other[:10]
random.shuffle(contexts3)
context3 = '\n\n'.join(contexts3)
with open(f"data/context.{len(context3)}.txt", 'w') as f:
    f.write(context3)