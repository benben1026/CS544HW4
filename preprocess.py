#!/usr/bin/env python

import sys, fileinput
import tree

for line in fileinput.input():
    t1 = tree.Tree.from_str(line)

    # Binarize, inserting 'X*' nodes.
    #t.binarize()
    t1.binarize_with_markovization()

    # Remove unary nodes
    t1.remove_unit()

    # The tree is now strictly binary branching, so that the CFG is in Chomsky normal form.

    # Make sure that all the roots still have the same label.
    assert t1.root.label == 'TOP'

    print t1


    t2 = tree.Tree.from_str(line)
    t2.remove_unit()
    t2.binarize()
    t2.parent_annotation()

    assert t2.root.label == 'TOP'
    print t2