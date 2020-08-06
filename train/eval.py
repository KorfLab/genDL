#!/usr/bin/env python3

import math

def summary_metrics(results):
	table = ""
	m = len(results.keys())
	
	TPR=0
	TNR=0
	PPV=0
	NPV=0
	FSC=0
	
	for k in results.keys():
		TPR += results[k]['TPR']
		TNR += results[k]['TNR']
		PPV += results[k]['PPV']
		NPV += results[k]['NPV']
		FSC += results[k]['FSC']
	
	TPR /= m
	TNR /= m
	PPV /= m
	NPV /= m
	FSC /= m
	
	TPRs = 0
	TNRs = 0
	PPVs = 0
	NPVs = 0
	FSCs = 0
	
	for k in results.keys():
		TPRs += (TPR-results[k]['TPR'])**2
		TNRs += (TNR-results[k]['TNR'])**2
		PPVs += (PPV-results[k]['PPV'])**2
		NPVs += (NPV-results[k]['NPV'])**2
		FSCs += (FSC-results[k]['FSC'])**2
	
	TPRs = math.sqrt(TPRs/m)
	TNRs = math.sqrt(TNRs/m)
	PPVs = math.sqrt(PPVs/m)
	NPVs = math.sqrt(NPVs/m)
	FSCs = math.sqrt(FSCs/m)
		
	table += "Metric | Avg    | Std    |\n"
	table += "------ | ------ | ------ |\n"
	table += f"TPR    | {TPR:.4f} | {TPRs:.4f} |\n"
	table += f"TNR    | {TNR:.4f} | {TNRs:.4f} |\n"
	table += f"PPV    | {PPV:.4f} | {PPVs:.4f} |\n"
	table += f"NPV    | {NPV:.4f} | {NPVs:.4f} |\n"
	table += f"FSC    | {FSC:.4f} | {FSCs:.4f} |\n"
	
	return table
	
def summary(results, time):
	
	report = ""
	m = len(results.keys())
	counter = 0
	top = dict()
	
	report += "All models\n"
	for k in sorted(results, key=lambda x: results[x]['FSC']):
		counter += 1
		if counter > 0.80*m: top[k] = results[k]
 
		report += f"{results[k]['model']}\n"
		report += f"TPR: {results[k]['TPR']:.4f}"
		report += f" TNR: {results[k]['TNR']:.4f}"
		report += f" PPV: {results[k]['PPV']:.4f}"
		report += f" NPV: {results[k]['NPV']:.4f}"
		report += f" FSC: {results[k]['FSC']:.4f}\n"
	
	report += "\n"
	report += f"Number of models: {m}\n"
	report += f"Total time: {time}\n"
	report += f"Avg time per model: {time/m:.4f}\n"
	report += "\n"
	report += "Avg metrics over all models\n"
	report += summary_metrics(results)
	report += "\n"
	report += "Avg metrics over top 20% models\n"
	report += summary_metrics(top)
	report += "\n"
	
	return report