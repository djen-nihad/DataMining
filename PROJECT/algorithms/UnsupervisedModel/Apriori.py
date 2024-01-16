import numpy as np
import pandas as pd

from algorithms.Utile import is_subList, diffrence, is_subList_2


class Apriori:

    def __init__(self, data=None, min_sup=0.5, min_conf=0.5):
        self.data = np.array(data)
        self.min_sup = min_sup
        self.min_conf = min_conf

    def fit(self):
        self.createTransaction()
        Lk, sup = self.apriori()
        rules = self.calculerSupportRLK(Lk)
        df = self.printResult(rules)
        return df

    def createTransaction(self):
        self.transactions = self.data.tolist()
        self.items = self.data.reshape(-1, 1)
        self.items = np.array(self.items).ravel().astype(str)
        self.items = np.unique(self.items, return_counts=False).tolist()
        self.items = [[item] for item in self.items]

    def calculerSupport(self, K_Items):
        count = 0
        for transaction in self.transactions:
            if is_subList(K_Items, transaction):
                count += 1
        count = count / len(self.transactions)
        return count

    def generateLk(self, candidates):
        L = []
        sups = []
        refuseCandidates = []
        for candidate in candidates:
            sup = self.calculerSupport(candidate)
            if sup >= self.min_sup:
                L.append(candidate)
                sups.append(sup)
            else:
                refuseCandidates.append(candidate)
        return L, refuseCandidates, sups

    def generateCondidat(self, Lk, refuseCondidat):
        c = []
        for i in range(len(Lk) - 1):
            for j in range(i + 1, len(Lk)):
                candidate = Lk[i].copy()
                for k in range(len(Lk[j])):
                    item = Lk[j][k]
                    if item in Lk[i]:
                        continue
                    candidate.append(item)
                    if not is_subList_2(candidate, c) and not is_subList_2(candidate, refuseCondidat):
                        c.append(candidate)
        return c

    def apriori(self):
        Lk = []
        sup = []
        # genreate C1 et sup1 :
        c = self.items
        L, refuseC, s = self.generateLk(c)
        while len(L) != 0:
            Lk.extend(L)
            sup.extend(s)
            c = self.generateCondidat(L, refuseC)
            L, refuseC, s = self.generateLk(c)
        return Lk, sup

    def generateCombinations(self, list):
        l = len(list)
        list_Conbinaision = []
        levelOne = [[item] for item in list]
        list_Conbinaision.append(levelOne)
        for i in range(l - 1):
            level_i = []
            for item in list_Conbinaision[-1]:
                for item_l1 in levelOne:
                    item_copy = item.copy()
                    if not is_subList(item_l1, item_copy):
                        item_copy.extend(item_l1)
                        if not is_subList_2(item_copy, level_i):
                            level_i.append(item_copy)
            list_Conbinaision.append(level_i)
        return_list = []
        for k_items in list_Conbinaision:
            return_list.extend(k_items)
        return return_list

    def generateRuleLk(self, Lk):
        ruleLk = []
        for itemFreq in Lk:
            combination = self.generateCombinations(itemFreq)
            for i in range(len(combination)):
                condition = combination[i]
                for j in range(i + 1, len(combination)):
                    conclusion = combination[j]
                    if diffrence(condition, conclusion):
                        ruleLk.append([condition, conclusion])
                        ruleLk.append([conclusion, condition])
        return ruleLk

    def evaluerRegle(self, list_transactions, rule):
        A = rule[0]
        B = rule[1]
        A_U_B = A.copy()
        A_AND_B = []
        for item in B:
            if item not in A_U_B: A_U_B.append(item)
            if item in A: A_AND_B.append(item)
        supA = self.calculerSupport(A)
        supB = self.calculerSupport(B)
        supA_U_B = self.calculerSupport(A_U_B)
        supA_AND_B = self.calculerSupport(A_AND_B)
        support = supA_U_B / len(list_transactions)
        confidence = supA_U_B / supA
        lift = supA_U_B / (supA * supB)
        if confidence != 1:
            Conviction = (1 - supB) / (1 - confidence)
        else:
            Conviction = 0
        Cosinus = supA_AND_B / np.sqrt(supA * supB)
        return support, confidence, lift, Conviction, Cosinus

    def equal_listes(self, liste1, liste2):
        return is_subList(liste1, liste2) and is_subList(liste2, liste1)

    def equal_rules(self, rule1, rule2):
        return self.equal_listes(rule1[0], rule2[0]) and self.equal_listes(rule1[1], rule2[1])

    def calculerSupportRLK(self, Lk):
        rule_valide = []
        rules = self.generateRuleLk(Lk)
        for rule in rules:
            evaluation = self.evaluerRegle(self.transactions, rule)
            if evaluation[1] >= self.min_conf:
                rule_existe = False
                for rv in rule_valide:
                    if self.equal_rules(rv[0], rule):
                        rule_existe = True
                        break
                if not rule_existe:  rule_valide.append((rule, evaluation))
        return rule_valide

    def printResult(self, rules):
        if len(rules) == 0:
            return -1
        columns = ["Rule", "Confidence", "Lift", "Convictions", "Cousins"]
        list_ruls = []
        df = pd.DataFrame(columns=columns)
        max_length = len(rules[-1][0][1]) + 1
        for length in range(max_length):
            for rule in rules:
                if len(rule[0][0]) != length: continue
                name_rule = ""
                for i in range(len(rule[0][0]) - 1):
                    name_rule += rule[0][0][i] + " AND "
                name_rule += rule[0][0][-1] + " ----> "
                for i in range(len(rule[0][1]) - 1):
                    name_rule += rule[0][1][i] + " AND "
                name_rule += rule[0][1][-1]
                line = {"Rule": name_rule, "Confidence": rule[1][1], "Lift": rule[1][2],
                        "Convictions": rule[1][2], "Cousins": rule[1][4]}
                list_ruls.append([name_rule, round(rule[1][1], 2), round(rule[1][2], 2),
                                  round(rule[1][3], 2), round(rule[1][4], 2)])
        return pd.DataFrame(list_ruls, columns=columns)
