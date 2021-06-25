import numpy
import csv
import pandas
import math

IS_SICK = 0

class TreeNode(object):
    def __init__(self) -> None:
        super().__init__()

        self.high_node = None
        self.low_node = None



class ID3:
    def fit_predict(self, train, test):
        patient_list = []
        for i in range(len(train)):
            patient_list.append(train[i].tolist())
            for j in range(1, len(train[i])):
                patient_list[i][j] = float(patient_list[i][j])

        properties = []
        for i in range(1, len(patient_list[0])):
            properties.append(i)

        root_node = self.recursiv_identefier(patient_list, properties)
        print("end")

    def recursiv_identefier(self, remaining_patients, remaining_properties):
        if self.check_if_homogeneous(remaining_patients):
            return TreeNode()  # TODO ? Entropy is 0
        if len(remaining_properties) == 0:
            return -1  # TODO

        curr_ig = 0
        best_ig = 0
        best_ig_edge = 0
        best_prop = remaining_properties[0]
        for prop in remaining_properties:

            total_sick = self.calc_num_of_sick(remaining_patients)
            nom_of_high_value_patients_sick = total_sick
            nom_of_low_value_patients_sick = 0
            remaining_patients.sort(key=lambda x: x[prop])

            for i in range(len(remaining_patients) - 1):
                if remaining_patients[i][0] == "M":
                    nom_of_high_value_patients_sick -= 1
                    nom_of_low_value_patients_sick += 1
                    curr_ig = self.calc_information_gain(len(remaining_patients), i,nom_of_low_value_patients_sick,
                                                              nom_of_high_value_patients_sick)
                    if curr_ig >= best_ig:
                        best_ig = curr_ig
                        best_ig_edge = (remaining_patients[i][prop] + remaining_patients[i + 1][prop]) / 2
                        best_prop = prop

        high_patient = []
        low_patient = []
        for patient in remaining_patients:
            if (patient[best_prop] < best_ig_edge):
                low_patient.append(patient)
            else:
                high_patient.append(patient)


        low_properties = remaining_properties.copy()
        high_properties = remaining_properties.copy()
        low_properties.remove(best_prop)
        high_properties.remove(best_prop)

        new_node = TreeNode()
        new_node.low_node = self.recursiv_identefier(low_patient, low_properties)
        new_node.high_node = self.recursiv_identefier(high_patient, high_properties)

        return new_node

    @staticmethod
    def check_if_homogeneous(patients):
        for i in range(len(patients) - 1):
            if patients[i][IS_SICK] != patients[i + 1][IS_SICK]:
                return False
        return True

    @staticmethod
    def calc_num_of_sick(patients):
        total_sick = 0
        for patient in patients:
            if patient[IS_SICK] == 'M':
                total_sick += 1
        return total_sick

    @staticmethod
    def calc_and_check_entropy(probability_m, probability_b):
        if probability_m == 0 or probability_b == 0:
            return 0

        return -(probability_m * math.log(probability_m, 2) +
                              probability_b * math.log(probability_b, 2))

    def calc_information_gain(self, num_of_patients, i, num_sick_low, num_sick_high):

        num_of_patients_low = i + 1
        num_of_patients_high = num_of_patients - num_of_patients_low

        probability_m_low = num_sick_low / num_of_patients_low
        probability_m_high = num_sick_high / num_of_patients_high
        probability_m_total = (num_sick_low + num_sick_high) / num_of_patients
        probability_b_total = 1 - probability_m_total
        probability_b_low = (num_of_patients_low - num_sick_low) / num_of_patients_low
        probability_b_high = (num_of_patients_high - num_sick_high) / num_of_patients_high

        parent_entropy = self.calc_and_check_entropy(probability_m_total, probability_b_total)
        child_low_entropy = self.calc_and_check_entropy(probability_m_low, probability_b_low)
        child_high_entropy = self.calc_and_check_entropy(probability_m_high, probability_b_high)
        information_gain = parent_entropy - (num_of_patients_low / num_of_patients) * child_low_entropy - \
            (num_of_patients_low / num_of_patients) * child_high_entropy
        return information_gain
