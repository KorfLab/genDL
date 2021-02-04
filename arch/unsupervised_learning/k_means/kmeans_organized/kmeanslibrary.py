#kmeans
import argparse
import yaml
import fileinput
import random
import pandas as pd
from sklearn.cluster import KMeans
import sys
import pprint
import math


def yaml_file_create_sequences(file1, file2, file3, minimum, maximum, name, num):

    dict_k_means = {}

    value1 = []
    with open((file1),"r") as file:
        key1 = file1.replace(".txt","")

        line = file.read().splitlines()
        random.shuffle(line)
        for i in range(num):
            value1.append(line[i][minimum:maximum])
    dict_k_means[key1] = value1

    if file2 is not None:
        value2 = []
        with open((file2),"r") as file:
            key2 = file2.replace(".txt","")

            line = file.read().splitlines()
            random.shuffle(line)
            for i in range(num):
                value2.append(line[i][minimum:maximum])

        dict_k_means[key2] = value2

    if file3 is not None:
        value3 = []
        with open((file3),"r") as file:
            key3 = file3.replace(".txt","")

            line = file.read().splitlines()
            random.shuffle(line)
            #print(len(line))
            for i in range(num):
                value3.append(line[i][minimum:maximum])

        dict_k_means[key3] = value3


    new = yaml.dump(dict_k_means)

    with open(name+'.yaml', 'w') as f:
        data = yaml.dump(dict_k_means, f)

    return(data)



def kmeans(yaml_file, clusters):
    sequences = []
    keys = []

    with open(yaml_file) as f:
        information = yaml.load_all(f, Loader=yaml.FullLoader)
        for i in information:
            for k, s in i.items():
                keys.append(k)
                sequences.append(s)

    ###conveting sequences from list of sublists into 1D list

    one_dim_table =[]
    for i in sequences:
        one_dim_table += i
    assert(clusters <= len(one_dim_table))


    ###converting sequences to their equivalent numerical value in order to calculate kmeans
    list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

    converted_sequences = []
    ###change it later
    each_seq_as_a_sublist = []
    ###

    for i in range(len(one_dim_table)):
        converted_sequences.append([])
        ###
        each_seq_as_a_sublist.append([])

    for item in range(len(one_dim_table)):
        #print(item)
        for i in (one_dim_table[item]):
            #print(item[i])
            if i in list_bases.keys():
                #print(i)
                converted_sequences[item].append(list_bases[i])
                each_seq_as_a_sublist[item].append(i)

            #------
            ###perform it only with trues
    ###formating converted sequences using pandas to later pass it to kmeans
    df = pd.DataFrame(converted_sequences)
    #df = df.transpose()

    headers = []
    for i in range(len(converted_sequences[0])):
        headers.append(str(f'p{i}'))
    df.columns = headers


    ###performing k_means clustering, input is the dataframe created above
    kmeans = KMeans(clusters).fit(df)
    centroids = kmeans.cluster_centers_

    #checking_percent_error(kmeans.labels_, converted_sequences, keys)


    return (#print('Labels:', kmeans.labels_, '\n', 'Centroids:','\n', centroids),
            #checking_percent_error(kmeans.labels_, converted_sequences, keys),
            comparing_sequences_with_the_same_label(clusters, kmeans.labels_, each_seq_as_a_sublist, keys))
            #labels_to_all_the_sequences(kmeans.labels_, each_seq_as_a_sublist, keys))



def checking_percent_error(labels, sequences, names_of_files):
    array_div = len(sequences)//len(names_of_files)

    nested_dict = {}

    for i in range(len(names_of_files)):
        nested_dict[names_of_files[i]] = {}
        analyzing = labels[i*array_div:((i+1)*array_div)]

        #print(analyzing, len(analyzing))
        for checking_label in analyzing:
            if checking_label in nested_dict[names_of_files[i]]:
                nested_dict[names_of_files[i]][checking_label] += 1
            else:
                nested_dict[names_of_files[i]][checking_label] = 1

    return ((nested_dict))


def labels_to_all_the_sequences(labels, sequences, names_of_files):

    labels_for_all_seq = {}


    for label, sequence in zip(labels, sequences):
        #i = 0
        sequence = ''.join(sequence)
        #print((label), (sequence))

        if label in labels_for_all_seq:
            labels_for_all_seq[label].append(sequence)
        else:
            labels_for_all_seq[label] = []
            labels_for_all_seq[label].append(sequence)

    neat_labels_dict = pprint.pformat(labels_for_all_seq)

    return ((neat_labels_dict))
    ###gc_and_at_content
    ###edit distance

def comparing_sequences_with_the_same_label(clusters, labels, sequences, names_of_files):

    labels_splitted_by_file = {}

    array_div = len(sequences)//len(names_of_files)

    for i in range(len(names_of_files)):
        labels_splitted_by_file [names_of_files[i]] = {}
        analyzing_labels = labels[i*array_div:((i+1)*array_div)]
        analyzing_seq = sequences[i*array_div:((i+1)*array_div)]
        #print(len(analyzing_seq), len(analyzing_seq))


        #print(names_of_files[i])
    #print(sequences)
        for label, sequence in zip(analyzing_labels, analyzing_seq):
            sequence = ''.join(sequence)
            #print(label, sequence)
            if label in labels_splitted_by_file[names_of_files[i]]:
                #print(labels_splitted_by_file[names_of_files[i]])
                labels_splitted_by_file[names_of_files[i]][label].append(sequence)
                #print('yes')
            else:
                labels_splitted_by_file[names_of_files[i]][label] = []
                labels_splitted_by_file[names_of_files[i]][label].append(sequence)


    neat_labels_splitted_byfile = pprint.pformat(labels_splitted_by_file)


    return (#print(neat_labels_splitted_byfile),
            generating_consensus_sequence(number_of_clusters = clusters, data_labeled_sequences = labels_splitted_by_file))



def generating_consensus_sequence(number_of_clusters, data_labeled_sequences):
    for name_of_file, label in data_labeled_sequences.items():
        if 'true' in name_of_file:
            consensus_dict = {}
            pfm_dict = {}
            for cluster_label, list_of_seq in label.items():
                storing_bases = []
                for i in range(len(list_of_seq[0])):
                    storing_bases.append([])

                for sequence in list_of_seq:
                    for base in range(len(sequence)):
                        storing_bases[base].append(sequence[base])
                #print(storing_bases, len(storing_bases), len(storing_bases[0]))

                converted_sequences = []
                pfm = []
                for i in range(len(list_of_seq[0])):
                    converted_sequences.append([])
                    pfm.append([])


                for bases in range(len(storing_bases)):
                    profile = {'A':0, 'C':0 , 'G':0, 'T':0}

                    for base in storing_bases[bases]:
                        if base in profile:
                            profile[base] += 1
                    #print(profile)
                    m = max(profile.values())


                    i = 0
                    for letter, frequency in profile.items():
                        if frequency == m:
                            converted_sequences[bases].append(letter)
                        pfm[bases].append(frequency)
                        #print(letter, frequency)
                #print(pfm)

                #print(converted_sequences, len(converted_sequences))


                consensus = []
                nucleic_acid_code ={'A':'A', 'C':'C', 'G':'G', 'T':'T',
                                    'CT':'Y', 'AG':'R', 'ACGT':'N',
                                    'GT':'K', 'AC':'M', 'CG':'S',
                                    'AT':'W', 'CGT':'B', 'AGT':'D',
                                    'ACT':'H', 'ACG':'V'}
                for nucleotides in converted_sequences:
                    nucleotides = ''.join(sorted(nucleotides))
                    #print(nucleotides)
                    if nucleotides in nucleic_acid_code:
                        consensus.append(nucleic_acid_code[nucleotides])
                consensus_dict[cluster_label] = ''.join(consensus)

                pfm_dict[cluster_label] = pfm

    return (consensus_dict, pfm_dict), position_weight_matrix(pfm_dict)



def position_weight_matrix(pfm): #ppm
    storage = {}
    for label, frequency in pfm.items():

        pwm = []
        for i in range(len(frequency)):
            pwm.append([])

        for pwm_list in range(len(frequency)):
            total_number_of_bases = sum(frequency[pwm_list])
            #print(total_number_of_bases, frequency[pwm_list])

            for pwm_value in frequency[pwm_list]:
                #print(pfm_value)
                try:
                    calculation = (pwm_value/total_number_of_bases)/0.25
                    caclulated_value = round((math.log(calculation, 2.0)),5)
                    #print(caclulated_value)
                    pwm[pwm_list].append(caclulated_value)

                except ValueError:
                    pwm[pwm_list].append((math.inf))
        storage[label] = pwm

    return print(storage)

def position_probability_matrix(pfm):
    storage = {}


    for label, frequency in pfm.items():

        ppm = []
        for i in range(len(frequency)):
            ppm.append([])

        for ppm_list in range(len(frequency)):
            total_number_of_bases = sum(frequency[ppm_list])

def graph():
    #using heat_map
    data = go.Heatmap()
    return



