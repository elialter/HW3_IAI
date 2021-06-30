import numpy
import csv
import pandas
import math
from sklearn.model_selection import KFold

IS_SICK = 0


class TreeNode(object):
    def __init__(self) -> None:
        super().__init__()
        self.sickness = None
        self.property = None
        self.edge_value = None
        self.high_node = None
        self.low_node = None


class ID3:

    def fit_predict(self, train, test, minimum_items_to_split=2):

        patient_list = []
        for i in range(len(train)):
            patient_list.append(train[i].tolist())
            for j in range(1, len(train[i])):
                patient_list[i][j] = float(patient_list[i][j])

        test_list = []
        for i in range(len(test)):
            test_list.append(test[i].tolist())
            for j in range(1, len(test[i])):
                test_list[i][j] = float(test_list[i][j])
        test = test_list

        properties = []
        for i in range(1, len(patient_list[0])):
            properties.append(i)

        root_node = self.recursive_identifier(patient_list, properties, minimum_items_to_split)
        new_test = []
        for i in range(len(test)):
            node = root_node
            while node.high_node is not None or node.low_node is not None:
                if test[i][node.property - 1] >= node.edge_value:
                    node = node.high_node
                else:
                    node = node.low_node
            if node.sickness == "M":
                new_test.append(1)
            else:
                new_test.append(0)

        array = numpy.array(new_test)
        return array

    def recursive_identifier(self, remaining_patients, properties, minimum_items_to_split):

        if self.check_if_homogeneous(remaining_patients):
            leaf = TreeNode()
            if remaining_patients[0][IS_SICK] == 'M':
                leaf.sickness = 'M'
            else:
                leaf.sickness = 'B'
            return leaf

        if len(remaining_patients) < minimum_items_to_split:
            leaf = TreeNode()
            leaf.sickness = self.sickness_majority(remaining_patients)
            return leaf

        best_ig = -1
        best_ig_edge = 0
        best_prop = properties[0]
        for prop in properties:

            total_sick = self.calc_num_of_sick(remaining_patients)
            nom_of_high_value_patients_sick = total_sick
            nom_of_low_value_patients_sick = 0
            remaining_patients.sort(key=lambda x: x[prop])

            for i in range(len(remaining_patients) - 1):
                if remaining_patients[i][0] == "M":
                    nom_of_high_value_patients_sick -= 1
                    nom_of_low_value_patients_sick += 1
                # If the higher value equals to the lower value, then we don't want to separate them by this
                # property, otherwise we won't be consistent.  We will pick a different way to separate them later on (
                # choosing a different property).
                if are_not_equal(remaining_patients[i][prop], remaining_patients[i + 1][prop]):
                    curr_ig = self.calc_information_gain(len(remaining_patients), i, nom_of_low_value_patients_sick,
                                                         nom_of_high_value_patients_sick)
                    if curr_ig >= best_ig:
                        best_ig = curr_ig
                        best_ig_edge = (remaining_patients[i][prop] + remaining_patients[i + 1][prop]) / 2
                        best_prop = prop

        high_patient = []
        low_patient = []
        for patient in remaining_patients:
            if patient[best_prop] < best_ig_edge:
                low_patient.append(patient)
            else:
                high_patient.append(patient)

        new_node = TreeNode()
        new_node.property = best_prop
        new_node.edge_value = best_ig_edge
        new_node.low_node = self.recursive_identifier(low_patient, properties, minimum_items_to_split)
        new_node.high_node = self.recursive_identifier(high_patient, properties, minimum_items_to_split)

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
                           (num_of_patients_high / num_of_patients) * child_high_entropy
        return information_gain

    def experiment(self, array, minimum_items_to_split):
        sets = KFold(n_splits=5, shuffle=True, random_state=311153746)
        total_error = 0
        for train_index, test_index in sets.split(array):
            test_index_list = test_index.tolist()
            training = []
            testing = []
            for i in range(len(array)):
                if i in test_index_list:
                    testing.append(array[i].tolist())
                else:
                    training.append(array[i].tolist())
            for row in testing:
                del row[0]
            training = numpy.array(training)
            testing = numpy.array(testing)
            result = self.fit_predict(training, testing, minimum_items_to_split)
            counter = 0
            j = 0
            for i in range(len(array)):
                if i in test_index_list:
                    if (result[j] == 1 and array[i][IS_SICK] == 'M') or (result[j] == 0 and array[i][IS_SICK] == 'B'):
                        counter += 1
                    j += 1
            total_error += (len(test_index_list) - counter)/len(test_index_list)
        error_average = total_error/5
        print(error_average)

    @staticmethod
    def sickness_majority(patients):
        sick_patients = 0
        healthy_patients = 0
        for row in patients:
            if row[IS_SICK] is 'M':
                sick_patients += 1
            else:
                healthy_patients += 1
        if sick_patients >= healthy_patients:
            return 'M'
        else:
            return 'B'


def are_not_equal(num_a, num_b):
    if num_a - num_b < 0.00001 and num_b - num_a < 0.00001:
        return False
    else:
        return True
