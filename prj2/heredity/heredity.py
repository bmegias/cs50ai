import csv
import itertools
import sys
import numpy

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")

def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    partial_results = []
    for name, info in people.items():
        gene_index = 1 if name in one_gene else 2 if name in two_genes else 0
        trait_index = name in have_trait
        if info['mother'] == None:
            prob = PROBS['gene'][gene_index] * PROBS['trait'][gene_index][trait_index]
            #print (f"{name}: {PROBS['gene'][gene_index]} * {PROBS['trait'][gene_index][trait_index]} = {prob}")
            partial_results.append(prob)
        else:
            mother = info['mother']
            m_gene_index = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father = info['father']
            f_gene_index = 1 if father in one_gene else 2 if father in two_genes else 0
            
            mu = PROBS['mutation']
            in_rules = {0: mu, 1: 0.5, 2: 1-mu}
            
            prob_m = in_rules[m_gene_index]
            prob_f = in_rules[f_gene_index]
            prob_temp = 0
            if gene_index == 1:
                prob_temp = (prob_m * (1-prob_f)) + (prob_f * (1-prob_m))
                #print (f"prob_temp =  ({prob_m} * {1-prob_f}) + ({prob_f} * {1-prob_m}) = {prob_temp}")
            elif gene_index == 2:
                prob_temp = prob_m * prob_f
                #print (f"prob_temp =  {prob_m} * {prob_f} = {prob_temp}")
            else:
                prob_temp = (1-prob_m) * (1-prob_f)
                #print (f"prob_temp =  {1-prob_m} * {1-prob_f} = {prob_temp}")              
                
            prob = prob_temp * PROBS['trait'][gene_index][trait_index]
            #print (f"{name}: {prob_temp} * {PROBS['trait'][gene_index][trait_index]} = {prob}")
            partial_results.append(prob)
            
    result = numpy.prod(partial_results)
    #print (result)
    return result
    #raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:
        gene_index = 1 if name in one_gene else 2 if name in two_genes else 0
        trait_index = name in have_trait
        probabilities[name]['gene'][gene_index] += p
        probabilities[name]['trait'][trait_index] += p
    return
    #raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        sum_gene = sum([v[1] for v in probabilities[name]['gene'].items()])
        for item in probabilities[name]['gene']:
            probabilities[name]['gene'][item] = probabilities[name]['gene'][item] / sum_gene
        sum_trait = sum([v[1] for v in probabilities[name]['trait'].items()])
        for item in probabilities[name]['trait']:
            probabilities[name]['trait'][item] = probabilities[name]['trait'][item] / sum_trait
    return
    #raise NotImplementedError


if __name__ == "__main__":
    main()
