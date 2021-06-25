import numpy
import csv
import pandas

IS_SICK = 0

class ID3:

 #   def fit_predict(self, train, test):
  #      if ()
   #
    #    calc current_entopy
     #
      #  for each property (f_i):
#
 #           for each edge value:
#
 #               calc left son's entropy
  #              calc right son's entropy
   #             if both are 1
    #                return f_i,edge_value
     #           else if this is the highest gane then keep f_i,edge_value as best
      #
       #     save best

    def recursiv_identefier(self, remaining_patients, remaining_properties,  propNum):
        if (self.check_if_homogeneous(remaining_patients)):
            return 1  # Entropy is 1

        for prop in remaining_properties:
            values = []
            for patient in remaining_patients:
                values += remaining_patients[patient][prop]
            values.sort()

            total_sick = calc_num_of_sick(remaining_patients)

            nom_of_high_value_patients_sick = total_sick
            nom_of_low_value_patients_sick = 0
            remaining_patients.sort(key=lambda x: x[prop])
            list_name.sort(key=lambda x: x[1], reverse=True)

            for i in len(values) - 1:
                for patient in remaining_patients:
                    if remaining_patients[patient][prop] >= (values[i] + (values[i] + 1)) / 2 :
                        high_value_patients.append(remaining_patients[patient])
                    else:
                        low_value_patients.append(remaining_patients[patient])
                above_edge_entopy = calc_entropy(high_value_patients)
                below_edge_entopy = calc_entropy(low_value_patients)
                high_value_patients.sort()
                low_value_patients.clear()


    def check_if_homogeneous(self, patients):
        for i in len(patients) - 1:
            if patients[i][IS_SICK] != patients[i + 1][IS_SICK]:
                return False
        return True

    def calc_entropy(self):
        return 2

    def calc_num_of_sick(self, patients):
        total_sick = 0
        for patient in patients:
            if patient[IS_SICK] == 'M':
                total_sick += 1
        return total_sick
