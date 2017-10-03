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
        # self.node_type = self.check_type(val) # 0 -> Nonterminals; 1 -> Terminals
        self.left = None
        self.right = None

    # def check_type(self, val):
    # 	if val == "<unk>":
    # 		return 1
    # 	elif val[-1] >= 'a' and val[0] <= 'z':
    # 		return 1
    # 	elif val.find("_") or val[-1] == '*' or (val[-1] >= 'A' and val[-1] <= 'Z') :
    # 		return 0
    # 	else: #punctuation
    # 		return 1


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

    def get_new_tag_dict(self, x, y, pro, bt=0):
        return {'LHS':x, 'RHS':y, 'pro': pro, 'bt': bt}

    def parse(self, text):
        tokens = text.split(" ")
        table = [[[] for j in xrange(len(tokens) + 1)] for i in xrange(len(tokens))]

        for offset in xrange(1, len(tokens) + 1):
            for i in xrange(len(tokens) - offset + 1):
                if offset == 1:
                    tmp = self.lookup(tokens[i])
                    for (x, y, pro) in tmp:
                        table[i][i + offset].append(self.get_new_tag_dict(x, y, pro))
                for mid in xrange(i + 1, i + offset):
                    if not table[i][mid] or not table[mid][i + offset]:
                        continue
                    for left in table[i][mid]:
                        for right in table[mid][i + offset]:
                            tmp = self.lookup(left['LHS'] + " " + right['LHS'])
                            if not tmp:
                                continue
                            for (x, y, pro) in tmp:
                                table[i][i + offset].append(self.get_new_tag_dict(x, y, pro + left['pro'] + right['pro'], mid))





        #table = [[[] for j in xrange(len(tokens) + 1)] for i in xrange(len(tokens))]
        # table = [[{'best': -sys.maxint - 1, 'tags': []} for j in xrange(len(tokens) + 1)] for i in xrange(len(tokens))]
        # backtrack = {}
        # for offset in xrange(1, len(tokens) + 1):
        #     for i in xrange(len(tokens) - offset + 1):
        #         if offset == 1:
        #             tmp = self.lookup(tokens[i])
        #             for (x, y, pro) in tmp:
        #                 table[i][i + offset]['tags'].append(x)
        #                 if pro > table[i][i + offset]['best']:
        #                     table[i][i + offset]['best'] = pro
        #                     backtrack[(i, i + offset)] = (x, y, 0)
        #             continue
        #         for mid in xrange(i + 1, i + offset):
        #             if not table[i][mid]['tags'] or not table[mid][i + offset]['tags']:
        #                 continue
        #             l_best = table[i][mid]['best']
        #             r_best = table[mid][i + offset]['best']
        #             for l_tag in table[i][mid]['tags']:
        #                 for r_tag in table[mid][i + offset]['tags']:
        #                     tmp = self.lookup(l_tag + " " + r_tag)
        #                     if not tmp:
        #                         continue
        #                     for (x, y, pro) in tmp:
        #                         table[i][i + offset]['tags'].append(x)
        #                         if pro + r_best + l_best > table[i][i + offset]['best']:
        #                             table[i][i + offset]['best'] = pro
        #                             backtrack[(i, i + offset)] = (x, y, mid)

        if not table[0][-1]:
            return ""
        return self.recursive_print_tree(0, len(tokens), table)
        #print self.rules["ADVP_RB"]
        # print table[0][1]
        # print table[0][-1]
        #print table
        #pretty(backtrack)
        #print self.recursive_print_tree(table, backtrack, 0, len(tokens))

    def lookup(self, items):
        output = []
        if items in self.reverse_rules:
            for i in self.reverse_rules[items]:
                output.append((i, items, math.log(self.reverse_rules[items][i], 10)))
        elif len(items.split(" ")) == 1:
            for i in self.reverse_rules["<unk>"]:
                output.append((i, items, math.log(self.reverse_rules["<unk>"][i], 10)))
        return output


    def recursive_print_tree(self, l, r, table, tag_chose = ""):
        if l + 1 == r:
            max_pro = -sys.maxint - 1
            best_tag = {}
            for tag in table[l][r]:
                if tag['pro'] > max_pro:
                    max_pro = tag['pro']
                    best_tag = tag
            return "(" + best_tag['LHS'] + " " + best_tag['RHS'] + ")"
        max_pro = -sys.maxint - 1
        best_tag = {}
        for tag in table[l][r]:
            if tag_chose != "" and tag['LHS'] != tag_chose:
                continue
            if tag['pro'] > max_pro:
                max_pro = tag['pro']
                best_tag = tag
        l_tag = best_tag['RHS'].split()[0]
        r_tag = best_tag['RHS'].split()[1]
        return "(" + best_tag['LHS'] + " " + self.recursive_print_tree(l, best_tag['bt'], table, l_tag) + " " + self.recursive_print_tree(best_tag['bt'], r, table, r_tag) + ")"



        # if l + 1 == r:
        #     return "(" + bt[(l, r)][0] + " " + bt[(l, r)][1] + ")"
        # mid = bt[(l, r)][2]
        # return "(" + bt[(l, r)][0] + " " + self.recursive_print_tree(l, mid, bt) + " " + self.recursive_print_tree(mid, r, bt) + ")"

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


lg = LearnGrammar()
with open('train.trees.pre.unk') as f:
    lines = f.readlines()
#for line in fileinput.input():
for line in lines:
    root = lg.recursion_build_tree(line)
    lg.parse_tree(root)

output_pro = open("rules", "w")
output_pro.write(lg.calculate_probability())
output_pro.close()


with open('dev.strings') as f:
    lines = f.readlines()
f_output = open('dev.parses', 'w')

#plot_file = open('length-time', 'w') 
text_length = []
elapsed_time = []
i = 1
p = Parser(lg.probability)
for line in lines:
    sys.stdout.flush()
    start_time = time.time()
    tmp = p.parse(line.strip())
    #length_time.append((len(line.split()), time.time() - start_time))
    text_length.append(len(line.split()))
    elapsed_time.append((time.time() - start_time) * 1000)
    f_output.write(tmp + "\n")

    print str(i) + " done."
    i += 1


plt.plot(text_length, elapsed_time, 'bo')
plt.axis([0, 20, 0, 200])
plt.show()

# p = Parser(lg.probability)
# print p.parse("The flight should be eleven a.m tomorrow .")
# print p.parse("I would like it to have a stop in New York and I would like a flight that serves breakfast .")


#unit test for lookup function
#print p.lookup("DT NP_NN")