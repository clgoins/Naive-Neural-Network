import matplotlib.pyplot as plt
import numpy as np


# Generates a list of dictionaries in the format {x:0, y:0}, where X and Y are random floating point values between the specifed minimums and maximums
def generateData(pointCount, minX, maxX, minY, maxY):
    data = []
    scaleX = maxX - minX
    scaleY = maxY - minY

    for i in range(pointCount):
        x = np.random.random()
        y = np.random.random()
        data.append({'x':x * scaleX + minX, 'y':y * scaleY + minY})

    return data


# Checks each item in the data list against a categorization function, and adds a label value; 1 if true, or 0 if false
def labelData(dataList):
    for point in dataList:
        if point['y'] >= categorize(point['x']):
            point['b'] = 1
            point['r'] = 0
        else:
            point['b'] = 0
            point['r'] = 1


def categorize(x):
    return 2 * x - 3


def visualizeData(dataList, showCatLine):
    xList = []
    yList = []
    cList = []
    largestValue = dataList[0]['x']
    lowestValue = largestValue

    for point in dataList:
        xList.append(point['x'])
        yList.append(point['y'])

        if showCatLine:
            if point['x'] > largestValue:
                largestValue = point['x']

            if point['y'] > largestValue:
                largestValue = point['y']

            if point['x'] < lowestValue:
                lowestValue = point['x']
            
            if point['y'] < lowestValue:
                lowestValue = point['y']
        
        if point['b'] == 1:
            cList.append([0,0,1])
        else:
            cList.append([1,0,0])

    plt.scatter(xList,yList, c=cList)

    if showCatLine:
        catLine = np.linspace(lowestValue,largestValue)
        plt.plot(catLine, categorize(catLine))


    plt.show()


def saveToCSV(dataList, filename):
    with open(filename,'w') as f:
        for point in dataList:
            f.write(f"{point['x']},{point['y']},{point['b']},{point['r']}\n")


def readFromCSV(filename):
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            split = line.split(',')
            data.append({'x':float(split[0]), 'y':float(split[1]), 'b':float(split[2]), 'r':float(split[3])})

    return data



for i in range(10):
    dataList = generateData(10000, -1000, 1000, -1000, 1000)
    labelData(dataList)
    saveToCSV(dataList, f"dataBIG{i}.csv")

visualizeData(dataList, True)