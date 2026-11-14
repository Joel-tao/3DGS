import unittest
from utils.scene_graph_local import build_hierarchy, query_hierarchy


def node(i,category,lo,hi):
    return dict(id=i,category=category,bounds_min_m=lo,bounds_max_m=hi,center_m=[(a+b)/2 for a,b in zip(lo,hi)])


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        self.table=node(1,'coffee table',[0,0,0],[2,2,1])
        self.cup=node(2,'cup',[.8,.8,1],[1,1,1.2])

    def test_supported_object_and_query(self):
        g=build_hierarchy([self.table,self.cup])
        result=query_hierarchy(g,'cup','coffee table')
        self.assertEqual([r['id'] for r in result],[2])
        self.assertEqual(result[0]['parent_id'],1)
        self.assertEqual(query_hierarchy(g, parent_id=1)[0]['id'],2)
        self.assertEqual(query_hierarchy(g, parent_id=99),[])
        self.assertEqual(result[0]['attachment_status'],'geometric_candidate')
        self.assertFalse(next(e for e in g['edges'] if e['source']==2)['verified'])

    def test_far_object_and_floating_object_are_unresolved(self):
        far=node(3,'book',[4,4,1],[4.2,4.2,1.2])
        floating=node(4,'cup',[.8,.8,3],[1,1,3.2])
        g=build_hierarchy([self.table,far,floating])
        for n in g['nodes'][2:]:
            self.assertEqual(n['parent'],'room')
            self.assertEqual(n['attachment_status'],'unresolved')

    def test_standalone_does_not_attach(self):
        window=node(3,'window',[.8,.8,.5],[1,1,1.2])
        g=build_hierarchy([self.table,window])
        self.assertEqual(g['nodes'][2]['object_type'],'Standalone')
        self.assertEqual(g['nodes'][2]['parent'],'room')

    def test_manual_classification_and_alias(self):
        chair=node(3,'sofa chair',[0,0,0],[1,1,1])
        g=build_hierarchy([chair],overrides={'3':{'object_type':'Standalone'}})
        self.assertEqual(query_hierarchy(g,'chair')[0]['object_type'],'Standalone')
        self.assertEqual(g['nodes'][1]['type_source'],'manual_override')
        with self.assertRaises(ValueError):
            build_hierarchy([chair],overrides={'3':{'object_type':'unknown'}})

    def test_every_instance_has_one_parent_and_no_cycles(self):
        g=build_hierarchy([self.table,self.cup,node(3,'pillow',[.2,.2,1],[.5,.5,1.1])])
        lookup={n['id']:n for n in g['nodes']}
        self.assertEqual(len(g['edges']),3)
        for n in g['nodes'][1:]:
            seen=set(); current=n['id']
            while current!='room':
                self.assertNotIn(current,seen)
                seen.add(current);current=lookup[current]['parent']


if __name__=='__main__':
    unittest.main()
