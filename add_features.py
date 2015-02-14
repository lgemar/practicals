import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as Descriptors
import csv

train_filename = 'test.csv'
feature_added = 'test_plus.csv'

csv_destination = open(feature_added, 'w')
with open(train_filename, 'r') as csv_handle: 
	reader = csv.reader(csv_handle)
	writer = csv.writer(csv_destination)
	next(reader, None)
	feature_list = []
	for row in reader: 
		smile = row[1]
		mol = Chem.MolFromSmiles(smile)
		# Place prior features in the feature list
		feature_list = [float(x) for x in row[2:258]]
		# Place new features in the list
		feature_list.append(Descriptors.CalcChi0n(mol))
		feature_list.append(Descriptors.CalcChi0v(mol))
		feature_list.append(Descriptors.CalcChi1n(mol))
		feature_list.append(Descriptors.CalcChi1v(mol))
		feature_list.append(Descriptors.CalcChi2n(mol))
		feature_list.append(Descriptors.CalcChi2v(mol))
		feature_list.append(Descriptors.CalcChi3n(mol))
		feature_list.append(Descriptors.CalcChi3v(mol))
		feature_list.append(Descriptors.CalcChi4n(mol))
		feature_list.append(Descriptors.CalcChi4v(mol))
		feature_list.append(Descriptors.CalcCrippenDescriptors(mol)[0])
		feature_list.append(Descriptors.CalcCrippenDescriptors(mol)[1])
		feature_list.append(Descriptors.CalcExactMolWt(mol)) 
		feature_list.append(Descriptors.CalcFractionCSP3(mol)) 
		feature_list.append(Descriptors.CalcHallKierAlpha(mol)) 
		feature_list.append(Descriptors.CalcKappa1(mol)) 
		feature_list.append(Descriptors.CalcKappa2(mol)) 
		feature_list.append(Descriptors.CalcKappa3(mol)) 
		feature_list.append(Descriptors.CalcLabuteASA(mol)) 
		feature_list.append(float(Descriptors.CalcNumAliphaticCarbocycles(mol))) 
		feature_list.append(float(Descriptors.CalcNumAliphaticHeterocycles(mol))) 
		feature_list.append(float(Descriptors.CalcNumAliphaticRings(mol))) 
		feature_list.append(float(Descriptors.CalcNumAmideBonds(mol))) 
		feature_list.append(float(Descriptors.CalcNumAromaticCarbocycles(mol))) 
		feature_list.append(float(Descriptors.CalcNumAromaticHeterocycles(mol))) 
		feature_list.append(float(Descriptors.CalcNumAromaticRings(mol))) 
		feature_list.append(float(Descriptors.CalcNumHBA(mol))) 
		feature_list.append(float(Descriptors.CalcNumHBD(mol))) 
		feature_list.append(float(Descriptors.CalcNumHeteroatoms(mol))) 
		# feature_list.append(float(Descriptors.CalcNumHeterocycles(mol))) 
		feature_list.append(float(Descriptors.CalcNumLipinskiHBA(mol))) 
		feature_list.append(float(Descriptors.CalcNumLipinskiHBD(mol))) 
		feature_list.append(float(Descriptors.CalcNumRings(mol)))
		feature_list.append(float(Descriptors.CalcNumRotatableBonds(mol)))
		feature_list.append(float(Descriptors.CalcNumSaturatedCarbocycles(mol)))
		feature_list.append(float(Descriptors.CalcNumSaturatedHeterocycles(mol)))
		feature_list.append(float(Descriptors.CalcNumSaturatedRings(mol)))
		feature_list.append(float(Descriptors.CalcTPSA(mol)))
		# Place gap information in the feature list
		# gap = float(row[257])
		# feature_list.append(gap)
		# Write to csv
		writer.writerow([smile] + feature_list)

csv_destination.close()
