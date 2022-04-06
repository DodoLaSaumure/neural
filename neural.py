import math
import random
from sklearn.utils import shuffle

LEARNINGRATE = 0.3
alpha = LEARNINGRATE
lambd = 0.1

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_prime(x):
    """Dérivée de la fonction sigmoid."""
    return sigmoid(x) * (1.0 - sigmoid(x))


class Cell():
    def __init__(self,inputs):
        self.inputs = inputs
        self.input_size = len(self.inputs)
        self.initBiasWeights()
    def setInput(self,inputs):
        self.inputs = inputs
#         self.calcOutput()
    def initBiasWeights(self):
        self.weights = []
        self.OlddeltaW= []
        for _ in range(self.input_size):
            self.weights.append(random.random())
            self.OlddeltaW.append(0.)
        self.bias = random.random()*2.-1.
        self.oldbias = 0.
    def calcOutput(self) :
        somme =0
        for index in range(self.input_size) :
            somme += self.inputs[index]*self.weights[index]
        somme += self.bias
        self.aggregation_z = somme
        self.activation_a = sigmoid(self.aggregation_z)

class Layer():
    def __init__(self,layerSize,layerInputList):
        self.layerSize = layerSize
        self.layerInputList=layerInputList
        self.input_size = len(layerInputList)
        self.cells = []
        self.createCells()
        self.aggregationList = []
        self.activationList = []
        self.calcOutput()
        
    def createCells(self):
        for _ in range(self.layerSize):
            cell = Cell(self.layerInputList)
            self.cells.append(cell)
    def calcOutput(self):
        self.aggregationList =[]
        self.activationList = []
        for cell in self.cells :
            cell.calcOutput()
            self.aggregationList.append(cell.aggregation_z)
            self.activationList.append(cell.activation_a)
    def setInput(self,layerInputList):
        self.layerInputList=layerInputList
        for index in range(len(self.cells)):
            self.cells[index].setInput(layerInputList)
        self.calcOutput()
        
class Network():
    def __init__(self, networkInput):
        self.layers = []
        self.samples = []
        self.networkInput = networkInput
    def addSample(self,sample):
        self.samples.append(sample)
    
    def trainSamples(self,batchSize,nbLearn):
        for _ in range(nbLearn) :
            newSamples = shuffle(self.samples)
            batch = newSamples[0:batchSize]
            for sample in batch:
                [input,expected]=sample
                self.backProp(input, expected)

    def addLayer(self,layerSize):
        if len(self.layers)< 1:
            layer = Layer(layerSize,self.networkInput)
        else :
            layer = Layer(layerSize,self.layers[-1].activationList)
        self.layers.append(layer)
        
    def calcOutput(self,networkInput):
        self.layers[0].setInput(networkInput)
        if len( self.layers)>1:
            for index  in range(1,len(self.layers)):
                self.layers[index].setInput(self.layers[index-1].activationList)

    def backProp(self,inputList,ExpectedList):
        deltasLists = [] # contient l'ensemble des listes de deltas
        self.calcOutput(inputList)
        computedOutputList = self.layers[-1].activationList
        self.accuracy = self.getAccuracy(ExpectedList,computedOutputList)
        # on commence par calculer l erreur sur la dernière couche
        DeltasOutput = self.getDeltasList(ExpectedList,computedOutputList)
        deltasLists.append(DeltasOutput)#la dernière couche sur l'index 0
        previousDeltasList = DeltasOutput # on la stocke car on en aura besoin pour rétropopager
        nb_layers = len(self.layers)
        for l in reversed(range(nb_layers - 1)):# on part de l'avant dernière couche, pour redescendre sur la couche 0
            deltasLayer = []
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            j = 0
            for cell in layer.cells :
                activation_prime = sigmoid_prime(cell.aggregation_z)
                deltaj = activation_prime*self.somme_sur_k_wkjdeltak(next_layer,previousDeltasList,j)
                deltasLayer.append(deltaj)
                j+=1
            previousDeltasList =deltasLayer 
            deltasLists.append(deltasLayer)#sur l'index 0 on a la dernière couche, et sur le dernier index on a la première couche
            
        deltasLists = list(reversed(deltasLists))# on remet la liste dans l'ordre des couches
        #wij(L) = wij(L) -alpha*ei(L)*aj(L-1)
        for l in range(nb_layers):
            layer = self.layers[l]
            for i in range(len(layer.cells)):
                for j in range(len(layer.cells[i].weights)):
                    #deltaW = -alpha * ei(L)            *aj(L-1)
                    deltaW = - alpha * deltasLists[l][i]*self.layers[l].cells[i].inputs[j]
                    layer.cells[i].weights[j]=layer.cells[i].weights[j]+ lambd*deltaW \
                    +(1-lambd)*layer.cells[i].OlddeltaW[j] #ajout d'inertie
                    layer.cells[i].OlddeltaW[j]=deltaW
                    deltaBias = - alpha* deltasLists[l][i]
                    layer.cells[i].bias = layer.cells[i].bias + lambd*deltaBias \
                    + (1-lambd)*layer.cells[i].oldbias
                    layer.cells[i].oldbias=deltaBias

    def somme_sur_k_wkjdeltak(self,next_layer,previousDeltasList,j):
        somme = 0.
        for k in range(len(next_layer.cells)):
            somme+= next_layer.cells[k].weights[j]*previousDeltasList[k]
        return somme

    def getAccuracy(self,expected,computed):
        somme =0.
        for index in range(len(computed)):
            somme+= (computed[index]-expected[index])**2
        somme =somme/(len(computed))
        somme = math.sqrt(somme)
        return somme

    def getDeltasList(self,expected,computed):
        deltasList = []
        for index in range(len(computed)):
            deltasList.append(computed[index]-expected[index])
        return deltasList
        
if __name__ == "__main__":
    network = Network([0.,0.])
    network.addLayer(2 )
    network.addLayer(3)
    network.addLayer(1)
#     outcell1 = network.layers[-1].cells[0]
# #         outcell2 = network.layers[-1].cells[1]
#     print (outcell1.activation_a,)# outcell2.activation_a)
#     print(network.layers[-1].activationList)
#     print("----")
    
    print ("back prop ")
    for sampleIndex in range(1000):
        a=random.random()/2.
        b=random.random()/2.
        inputSample = [a,b]
        outputsample = [a+b]
        network.addSample([inputSample,outputsample])
#         network.backProp(inputSample, outputsample)
#    network.trainSamples(100,100)
    print("acc",network.accuracy)
#     network.backProp([50,1,10], [2,2])
    print ("training done, now showing new cases")
    for sampleIndex in range (10):
        a=random.random()/2.
        b=random.random()/2.
        inputSample = [a,b]
        outputsample = [a+b]
#         network.calcOutput([50,1,10])
        print ("--->",a,b)
        network.calcOutput(inputSample)
        computed = network.layers[-1].activationList
        theorical = [a+b]
        print(network.layers[-1].activationList)
        print ([a+b])
        print ("Delta",computed[0]-theorical[0])
        print ("------")
