from collections import defaultdict
import pprint
import fileinput
import math
import sys
import matplotlib.pyplot as plt
import time


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class LearnGrammar:
    def __init__(self):
        self.probability = defaultdict(lambda: defaultdict(float))

    def parse_tree(self, root):
        if root.left == None:
            return
        if root.right == None:
            self.probability[root.val][root.left.val] += 1
            return
        self.probability[root.val][root.left.val + " " + root.right.val] += 1
        self.parse_tree(root.left)
        self.parse_tree(root.right)

    def calculate_probability(self):
        output = ""
        for row in self.probability:
            total = 0
            for i in self.probability[row]:
                total += self.probability[row][i]
            for j in self.probability[row]:
                self.probability[row][j] = self.probability[row][j] / float(total)
                output += row + " -> " + j + " # " + str(self.probability[row][j]) + "\n"
        return output

    def recursion_build_tree(self, text):
        token = self.tokenize(text)
        if not token:
            return None
        elif len(token) == 2:
            root = TreeNode(token[0])
            root.left = TreeNode(token[1])
            return root
        root = TreeNode(token[0])
        root.left = self.recursion_build_tree(token[1])
        root.right = self.recursion_build_tree(token[2])
        return root

    def tokenize(self, text):
        text = text.strip()[1:-1]
        pos = text.find("(")
        if pos == -1:
            return text.split(" ")
        output = [text[:pos].strip()]
        stack = []
        for i in xrange(len(text)):
            if text[i] == '(':
                stack.append(1)
            elif text[i] == ')' and stack:
                stack.pop(-1)
            elif text[i] == ')':
                print "Invalid input"
                return []
            else:
                continue

            if not stack:
                output.append(text[pos:i + 1].strip())
                output.append(text[i + 1:].strip())
                return output


class Parser:
    def __init__(self, rules):
        self.rules = rules
        self.reverse_rules = defaultdict(lambda: defaultdict(float))
        for i in self.rules:
            for j in self.rules[i]:
                self.reverse_rules[j][i] = self.rules[i][j]

    def parse(self, text, mode = 0):
        tokens = text.split(" ")
        best = {}
        back = {}
        for i in xrange(0, len(tokens)):
            for j in xrange(i + 1, len(tokens) + 1):
                best[(i,j)] = defaultdict(lambda: float(-sys.maxint - 1))
                back[(i,j)] = {}

        for i in xrange(1, len(tokens) + 1):
            tmp = self.lookup(tokens[i - 1])
            for (x, y, pro) in tmp:
                if pro > best[(i - 1, i)][x]:
                    best[(i - 1, i)][x] = pro
                    back[(i - 1, i)][x] = (x, y)
        for offset in xrange(2, len(tokens) + 1):
            for i in xrange(len(tokens) - offset + 1):
                j = i + offset
                for k in xrange(i + 1, j):
                    if not best[(i, k)] or not best[(k, j)]:
                        continue
                    for l_tag, l_pro in best[(i, k)].items():
                        for r_tag, r_pro in best[(k, j)].items():
                            tmp = self.lookup(l_tag + " " + r_tag)
                            for (x, y, pro) in tmp:
                                p = pro + l_pro + r_pro
                                if p > best[(i, j)][x]:
                                    best[(i, j)][x] = p
                                    back[(i, j)][x] = (x, y, k)
        if not back[(0, len(tokens))] and mode == 0:
            return ""
        elif not back[(0, len(tokens))]:
            output = []
            for i in xrange(len(tokens)):
                best_pro = -sys.maxint - 1
                best_tag = ""
                for t in best[(i, i + 1)]:
                    if best[(i, i + 1)][t] > best_pro:
                        best_pro = best[(i, i + 1)][t]
                        best_tag = t
                tag = best_tag
                if tag == "":
                    continue
                output.append("(" + back[(i, i + 1)][tag][0] + " " + back[(i, i + 1)][tag][1] + ")")
            return " ".join(output)
        return self.recursive_print_tree(0, len(tokens), best, back)    

    def lookup(self, items):
        output = []
        if items in self.reverse_rules:
            for i in self.reverse_rules[items]:
                output.append((i, items, math.log(self.reverse_rules[items][i], 10)))
        elif len(items.split(" ")) == 1:
            for i in self.reverse_rules["<unk>"]:
                output.append((i, items, math.log(self.reverse_rules["<unk>"][i], 10)))
        return output

    def recursive_print_tree(self, l, r, best, back, tag=None):
        if tag == None:
            best_pro = -sys.maxint - 1
            best_tag = ""
            for t in best[(l, r)]:
                if best[(l,r)][t] > best_pro:
                    best_pro = best[(l ,r)][t]
                    best_tag = t
            tag = best_tag
        if tag == None:
            return
        if l + 1 == r:
            return "(" + back[(l, r)][tag][0] + " " + back[(l, r)][tag][1] + ")"
        mid = back[(l, r)][tag][2]
        l_tag = back[(l, r)][tag][1].split()[0]
        r_tag = back[(l, r)][tag][1].split()[1]
        return "(" + back[(l, r)][tag][0] + " " + self.recursive_print_tree(l, mid, best, back, l_tag) + " " + self.recursive_print_tree(mid, r, best, back, r_tag) + ")"

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


lg_sibling = LearnGrammar()
lg_parent = LearnGrammar()
flag = True
for line in fileinput.input():
    if flag:
        root = lg_sibling.recursion_build_tree(line)
        lg_sibling.parse_tree(root)
        flag = not flag
    else:
        root = lg_parent.recursion_build_tree(line)
        lg_parent.parse_tree(root)
        flag = not flag

a = lg_sibling.calculate_probability()
b = lg_parent.calculate_probability()
# output_pro = open("rules_sibling", "w")
# output_pro.write(a)
# output_pro.close()


with open('dev.strings') as f:
    lines = f.readlines()
f_output = open('dev.parses', 'w')

#plot_file = open('length-time', 'w') 
text_length = []
elapsed_time = []
i = 1
p_sibling = Parser(lg_sibling.probability)
p_parent = Parser(lg_parent.probability)
for line in lines:
    # start_time = time.time()
    tmp = p_sibling.parse(line.strip())
    if tmp.strip() == "":
        tmp = p_parent.parse(line.strip(), 1)
    # text_length.append(math.log(len(line.split()), 10))
    # elapsed_time.append(math.log((time.time() - start_time) * 1000 * 300, 10))
    f_output.write(tmp + "\n")

    #print str(i) + " done."
    #i += 1
    #sys.stdout.flush()

# plt.figure(1)
# plt.plot(text_length, elapsed_time, 'bo')
# a = [0, 1, 1.2, 1.4, 1.6, 1.8, 2]
# plt.plot(a, [i * 3 for i in a])


# plt.axis([0, max(text_length), 0, max(elapsed_time)])
# plt.xlabel('Text Length (log)')
# plt.ylabel('Elapsed Time (log)')
# plt.show()

# p = Parser(lg.probability)
# print p.parse("The flight should be eleven a.m tomorrow .")
# print p.parse("I would like it to have a stop in New York and I would like a flight that serves breakfast .")


#unit test for lookup function
#print p.lookup("DT NP_NN")