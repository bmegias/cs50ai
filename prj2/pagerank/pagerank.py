import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    inv_damping = 1-damping_factor
    model = {p:0.0 for p in corpus.keys()}
    if len(corpus[page]) > 0:
        for p in corpus[page]:
            model[p] = damping_factor * 1/len(corpus[page])
    else:
        inv_damping = 1
    for p in corpus.keys():
        model[p] += inv_damping / len(corpus.keys())
    return model
    #raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus.keys())
    count = {p:0 for p in corpus.keys()}
    model = {p:1/N for p in corpus.keys()}
    
    for i in range(n):
        cur_page = random.choices(list(model.keys()), weights=model.values())[0]
        count[cur_page] += 1/n
        model = transition_model(corpus,cur_page,damping_factor)
    
    #print(model)
    return count
    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr = {}
    N = len(corpus.keys())
    for page in corpus.keys():
        pr[page] = 1/N
    
    fixed_corpus = {}
    for p in corpus:
        if not any(corpus[p]):
            fixed_corpus[p]=set(corpus.keys())
        else:
            fixed_corpus[p]=corpus[p]
    
    while True:
        new_pr = {}
        for p in pr.keys():
            pages_linking_to_p = [i for i in pr.keys() if p in fixed_corpus[i]]
            new_pr[p]= (1 - damping_factor) / N + damping_factor * sum([pr[i]/len(fixed_corpus[i]) for i in pages_linking_to_p])
        if not any([r for r in new_pr if abs(new_pr[r]-pr[r]) > 0.001]):
            break
        pr = new_pr
        #print(pr)
           
    return pr
    #raise NotImplementedError


if __name__ == "__main__":
    main()
