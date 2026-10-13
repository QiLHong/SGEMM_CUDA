#!/usr/bin/env python3
"""CPU coverage and bounded interleaving checks; not a substitute for Sanitizer."""
from functools import lru_cache
import unittest


def explore(overlap, tiles, single=False):
    producer=[]; consumer=[]
    for tile in range(tiles):
        if tile+1<tiles: producer.append(("store",tile+1))
        end=[] if single else [("barrier",tile,2)]
        producer += [("barrier",tile,1)]+end
        read=[("begin",tile),("end",tile)]
        first=[("barrier",tile,1)]
        consumer += (read+first if overlap else first+read)+end
    programs=[producer]+[consumer]*4

    @lru_cache(None)
    def visit(pcs, contents, readers):
        actions=[p[pc] if pc<len(p) else None for p,pc in zip(programs,pcs)]
        if all(a is None for a in actions):
            assert readers==(0,0)
            return
        if actions[0] and actions[0][0]=="barrier" and all(a==actions[0] for a in actions):
            visit(tuple(pc+1 for pc in pcs),contents,readers)
            return
        progressed=False
        for actor,action in enumerate(actions):
            if not action or action[0]=="barrier": continue
            progressed=True
            op,tile=action; slot=tile%2
            new_contents=list(contents); new_readers=list(readers); new_pcs=list(pcs)
            if op=="store":
                assert readers[slot]==0, "overwrite while a consumer is computing"
                new_contents[slot]=tile
            elif op=="begin":
                assert contents[slot]==tile, "read before ready / after overwrite"
                new_readers[slot]+=1
            else:
                assert contents[slot]==tile and readers[slot]>0
                new_readers[slot]-=1
            new_pcs[actor]+=1
            visit(tuple(new_pcs),tuple(new_contents),tuple(new_readers))
        assert progressed, "deadlock"
    visit((0,)*5,(0,-1),(0,0))
    return visit.cache_info().currsize


class K32Tests(unittest.TestCase):
    def test_producer_coverage(self):
        a=[(lane//4+t*8,(lane%4)*4+j) for lane in range(32)
           for t in range(16) for j in range(4)]
        b=[(k,lane*4+j) for lane in range(32) for k in range(16) for j in range(4)]
        self.assertEqual(len(set(a)),2048)
        self.assertEqual(set(a),{(m,k) for m in range(128) for k in range(16)})
        self.assertEqual(len(set(b)),2048)
        self.assertEqual(set(b),{(k,n) for k in range(16) for n in range(128)})
        self.assertEqual((16*132*4)%16,0)
        self.assertEqual((16*(132+128)*4)%16,0)

    def test_consumer_coverage(self):
        output=[]
        for tid in range(32,160):
            warp,lane=divmod(tid-32,32)
            row=64*(warp//2)+(2*(lane//16)+lane%2)*4
            col=64*(warp%2)+((lane%16)//2)*4
            output += [(row+(i//4)*16+i%4,col+(j//4)*32+j%4)
                       for i in range(16) for j in range(8)]
        self.assertEqual(len(output),128*128)
        self.assertEqual(len(set(output)),128*128)
        self.assertEqual(set(output),{(m,n) for m in range(128) for n in range(128)})

    def test_stage_lifetime(self):
        for overlap,single in ((False,False),(True,False),(True,True)):
            for tiles in (1,2,3,4):
                with self.subTest(overlap=overlap,single=single,tiles=tiles):
                    self.assertGreater(explore(overlap,tiles,single),0)


if __name__=="__main__": unittest.main()
