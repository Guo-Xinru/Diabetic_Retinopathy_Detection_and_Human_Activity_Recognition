import os
import numpy as np
import copy


class Data:
    def __init__(self):
        # public
        self.raw_data_norm = []
        self.label_guide = []
        self.label = []
        self.windowed_dataset = []
        self.windowed_labelset = []
        # private (temp data)
        self.windowed_expdata = []
        self.windowed_explabel = []
        # exp_2_windownumber store the coressponding exp id to the window numebr
        # used to read the windowed_dataset according to exp id
        self.exp_2_windownumber = []

    # slide window algorithmus
    def slide_window(self, window_size, window_shift_ratio, label_threshold):
        for exp_data, exp_label in zip(self.raw_data_norm, self.label):  # i col number and experiment id, row jetzige Col
            # exp_data, exp_label will go through with the experiment
            # check if the data and label paired
            assert len(exp_data) == len(exp_label), "data and label not paired"

            # total window number
            total_windows = (len(exp_data) - window_size) // int(window_size * window_shift_ratio) + 1

            # go through the total windows

            for i in range(total_windows):
                start_idx = i * int(window_size * window_shift_ratio)
                end_idx = start_idx + window_size
                # get the window values and window labels
                single_window_values = exp_data[start_idx:end_idx]
                # generate the new window label with voting algorithm
                single_window_labels = self.label_voting(exp_label[start_idx:end_idx], label_threshold)

                self.windowed_expdata.append(single_window_values)
                self.windowed_explabel.append(single_window_labels)

            # append the window of each exp to the array
            self.exp_2_windownumber.append(total_windows)

            # append every experiment to the dataset
            # append is shallow copy the list will still change in the dataset if the expdata change !!!!
            # hier is very important!!! very important !! with append
            self.windowed_dataset.append(copy.deepcopy(self.windowed_expdata))
            self.windowed_labelset.append(copy.deepcopy(self.windowed_explabel))

            # clear the windows label and data array to ready for next experiment
            # here ! should be noticed
            self.windowed_expdata.clear()
            self.windowed_explabel.clear()


    # label proccessing to decide the label of each window
    def label_voting(self, window_label, label_threshold):
        """
        :param window_label: Raw window label
        :param label_threshold: one   voting threshold
        # eg. 60%: threshold = 0.6
        # count appearance of all labels, 0..13:
        :return:
        """
        # 0-> max [0,X,X,X,max]
        label_appearance = np.bincount(window_label)
        # the most apperance label in the window label
        max_appearance = max(label_appearance)

        if max_appearance > label_threshold * len(window_label):
            predominant_label = np.argmax(label_appearance)
        else:
            predominant_label = 0

        """
        # look for full-transition-activity inside window and prioritize
        # some are very short compared to continuous actions, 
        # this is important because the continuous act is short
        Here we need to decrease the thereshold, because the transition is short
        """
        transition_threshold = label_threshold / 2
        if (len(label_appearance) > 7) and (predominant_label < 7):
            max_appearance_transitions = max(label_appearance[7:])

            if max_appearance_transitions > transition_threshold * len(window_label):
                # full transition-activity inside window OR big portion of window consists of transition-activity
                if ((window_label[0] < 7) and (window_label[-1] < 7)) \
                        or (max_appearance_transitions > (label_threshold - 0.2) * len(window_label)):
                    # the predominant label to the transition action
                    for i in range(len(label_appearance)):
                        if label_appearance[i] == max_appearance_transitions:
                            predominant_label = i

        predominant_label = np.eye(13, dtype=int)[predominant_label]
        return predominant_label

    # dataset packing function , devide to dataset
    def packing2_dataset(self, user_start_id, user_end_id):
        """
        the dataset should be devided with 70% / 20% / 10%
        self.label_guide = []
        self.windowed_dataset = []
        self.windowed_labelset = []
        :return:
        """
        exp_start_id = 99 # init value
        # finding the experiment id according to user id then handling the data
        for row in self.label_guide:
            if row[1] == user_start_id:
                if exp_start_id >= row[0]:
                    exp_start_id = row[0]
            if row[1] == user_end_id:
                exp_end_id = row[0]

        # convert the exp id to the position in windowed data
        start_line = sum(self.exp_2_windownumber[:exp_start_id-1])
        end_line = sum(self.exp_2_windownumber[exp_start_id:exp_end_id]) + start_line

        # select the daata and label according to the experiment id
        packing_data = self.windowed_dataset[exp_start_id:exp_end_id]
        packing_label = self.windowed_labelset[exp_start_id:exp_end_id]

        # flatten the 2D Array to 1D
        packing_data_flattened = [item for sublist in packing_data for item in sublist]
        packing_label_flattened = [item for sublist in packing_label for item in sublist]

        return packing_data_flattened, packing_label_flattened

