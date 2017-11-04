class Tester:
    def __init__(self):
        f = open("test.txt", 'r')
        self.trainingData = [line.split(",") for line in f.readlines()]
        f.close()
        for i in range(len(self.trainingData)):
            for j in range(len(self.trainingData[i])):
                self.trainingData[i][j] = int(self.trainingData[i][j].strip())
    def getTestSet(self):
        output = []
        for i in range(len(self.trainingData)):
            output.append(self.trainingData[i][0])
        return output

    def checkAnswers(self, answer):
        score = 0;
        for i in range(len(self.trainingData)):
            for j in range(len(answer[i])):
                if answer[i][j] in self.trainingData[i][1:]:
                    score += 1
        maxScore = 0
        for i in range(len(self.trainingData)):
            maxScore += len(self.trainingData[i]) - 1
        return score, maxScore
