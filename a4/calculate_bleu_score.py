r1 = "love can always find a way"
c1 = "the love can always do"

r2 = "love makes anything possible"
c2 = "love can make anything possible"

lambdas = [0.5,0.5,0,0]
import nltk
import numpy as np
def generate_ngrams(words,n):
    return [tuple(words[i:i+n]) for i in range(len(words) -n +1)]

def count_ngrams(s,t,n):
    s_words = s.split()
    t_words = t.split()
    if len(s_words) < n :
        return 0
    s_ngrams = generate_ngrams(s_words, n)
    t_ngrams = set(generate_ngrams(t_words,n))

    count = sum(1 for gram in s_ngrams if gram in t_ngrams)

    return count / len(s_ngrams) if s_ngrams else 0

def BP(r,c):
    l_r = len(r.split())
    l_c = len(c.split())
    print("lr lc: ",l_r, l_c)
    if l_c > l_r: 
        return 1
    else:
        return np.exp(1 - (l_r/l_c))

def bleu_score(r,c):
    p = [0]*4
    for n in range(2):
        p[n] = count_ngrams(c,r,n+1)
        print("p_n: ",p[n])
    temp = [np.log(p[i])*lambdas[i] if lambdas[i]>0 and p[i]>0 else 0 for i in range(2)]
    # unigram = count_ngrams(c,r,1)
    # bigram = count_ngrams(c,r,2)
    # print("unigram,bigram" ,unigram,bigram)
    # temp = [np.log(count_ngrams(c,r,1))*0.5,np.log(count_ngrams(c,r,2))*0.5]
    # print('temp: ',temp)
    bp = BP(r,c)
    print("bp: ",bp)
    BLEU_score = bp * np.exp(sum(temp))
    return BLEU_score



if __name__ == "__main__":

    print("BLEUscore: ",bleu_score(r1,c1))
    print("BLEUscore: ",bleu_score(r1,c2))
    # c1 better but c1 don't have intuitive sence => problem with bleu score
    print("BLEUscore: ",bleu_score(r2,c1))
    print("BLEUscore: ",bleu_score(r2,c2))
    # can have many candicate can true if have only one referennce => more ref can be good bleu score. 
