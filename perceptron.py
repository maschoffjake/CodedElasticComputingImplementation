import numpy as np
from os import listdir
from os.path import isfile, join

class Perceptron:

    # Set the weight vector to uniform randomized data between -0.01 and 0.01
    def __init__(self, lr = 0.01, decaying_learning_rate=False):
        self.t = 0
        self.learning_rate = lr

    # Train function for the Perceptron.
    def train(self, weights, training_sample, label):

        update = self.calculate_score(weights=weights, data=training_sample, label=label)

        if update:
            self.update_weights(data=training_sample, label=label)

    def update_weights(self, data, label):

        # Makes it easier for multiply by values
        if label == 0:
            label = -1

        # First update the bias
        self.W[0] += label * self.learning_rate * self.W[0]

        # Now update all the weights that were used in this examples
        for word_num in data:
            count = data[word_num]
            self.W[word_num] += label * self.learning_rate * count


    def calculate_score(self, data, label):

        # Start withe the bias term (first term in the weight matrix is the bias, so multiply it by 1)
        score = self.W[0]

        for word_num in data:
            count = data[word_num]
            weight = self.W[word_num]

            # Increase the score by the weight of the current word * count of the word in this review
            score += weight * count

        # If the label was -1, multiple it by negative 1 since you are supposed to multiply by the correct label
        if label == 0:
            score = score * -1


        # If the score is less than 0, then update!
        if score <= 0:
            return True
        return False

    def predict_score(self, data, label, average_perceptron=False):

        # Start withe the bias term (first term in the weight matrix is the bias, so multiply it by 1)
        score = 0
        for word_num in data:
            count = data[word_num]
            weight = self.W[word_num]

            # Increase the score by the weight of the current word * count of the word in this review
            score += weight * count

        # If the score is less than 0, then incorrect!
        if score <= 0:
            return 0
        return 1

    # Predict on the given test samples and labels and return the accuracy
    # Average_perceptron: default False, if given True it will use self.A weights to make predictions instead of self.W
    def predict(self, label, test_sample, average_perceptron= False, dev=False):

        predicted_label = self.predict_score(data=test_sample, label=label, average_perceptron=average_perceptron)

        if predicted_label == label:
            return 1
        return 0

        # return predicted_label

    # Perform a cross validation on all the files located in the directory provided
    # Learning_rate: default is 0.1, otherwise specify the learning rate that you would like to use while training
    # Decaying_learning_rate: default false, if given True, it will use learning_rate as initial learning rate and decay the learning rate every training sample
    # Margin: default None, if given a value, it must be correctly classified by at least that margin value otherwise it will perform an update as if it were a mistake
    # Average perceptron: default False, if given True it will also update the self.A weights to be used for the average perceptron classifier
    def cross_validation(self, directory, learning_rate=0.1, decaying_learning_rate= False, margin=None, average_perceptron=False):

        accuracy = 0

        # Grab all of the files from the directory
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        # Change the test set every iteration
        for test_set in range(len(onlyfiles)):

            # Reset weights every time you call cross validation
            self.reset_weights()

            # Load the test data
            test_samples, test_labels = load_samples_and_labels(directory + onlyfiles[test_set])

            for learning_set in range(len(onlyfiles)):

                # Do not learn on the test set
                if learning_set == test_set:
                    continue

                training_samples, labels = load_samples_and_labels(directory + onlyfiles[learning_set])

                self.train(training_samples=training_samples, labels=labels, epochs=10, learning_rate=learning_rate, decaying_learning_rate=decaying_learning_rate, margin=margin, average_perceptron=average_perceptron)


            accuracy += self.predict(test_samples=test_samples, test_labels=test_labels, average_perceptron=average_perceptron)

        return accuracy/len(onlyfiles)
