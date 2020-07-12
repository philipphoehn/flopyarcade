from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
import os
from os.path import join
from pickle import dump, load
from sklearn import preprocessing
import time
from time import sleep


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)


from FloPyArcade import FloPyEnv
from FloPyArcade import FloPyAgent

# wrkspc = 'C:\\FloPyArcade'
wrkspc = os.path.dirname(os.path.realpath(__file__))
recompileDataset = False
reTrain = True
reTest = True
modelName = 'surrogateEnv3_sigmoid'
# modelName = 'surrogateModelStatesWeighted2'
trainSamplesLimit = None
testSamplesLimit = None
learningRate = 0.00061
# learningRateInterval = [-3.8, -4.2]
learningRateInterval = [-1.0, -1.0]
batch_size = 128 # 32784
epochs = 1000000
patience = 1000

maxSamples = 200000
nRandomModels = 1

continueModelTrain = True



class FloPyEnvSurrogate():

    def __init__(self, name):
        self.name = name


    def compileDataset(self):
        pass




def unnormalizeInitial(data, env):
    from numpy import multiply
    data = multiply(data, env.maxH)

    return data

def calculate_weightsforregression(y, nbins=30):
    '''
    Calculate weights to balance a binary dataset
    '''
    from numpy import digitize, divide, histogram, ones, subtract, sum, take

    # hist, bin_edges = histogram(y, bins=nbins, density=True)
    # returning relative frequency
    hist, bin_edges, patches = plt.hist(y, bins=nbins, density=True)
    # returning true counts in bin
    # hist, bin_edges, patches = plt.hist(y, bins=nbins, density=False)
    # to receive indices 1 has to be subtracted
    freqIdx = subtract(digitize(y, bin_edges, right=True), 1)
    freqs = take(hist, freqIdx)

    if 0. in freqs:
        sample_weights = ones(len(freqs))
    else:
        # choosing 100 in numerator to avoid too small weights with biased datasets
        # potentially make this dependent on the magnitude difference between min and max?
        # print('min freq', min(freqs))
        sample_weights = divide(1., freqs)

    # print('min, max', min(sample_weights), max(sample_weights))

    return np.asarray(sample_weights)

# def GPUAllowMemoryGrowth():
#     """Allow GPU memory to grow to enable parallelism on a GPU."""
#     from tensorflow.compat.v1 import ConfigProto, set_random_seed
#     from tensorflow.compat.v1 import Session as TFSession
#     # from tensorflow.compat.v1.keras import backend as K
#     from tensorflow.compat.v1.keras.backend import set_session
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = TFSession(config=config)
#     # K.set_session(sess)
#     set_session(sess)


env = FloPyEnv()
FloPyAgent().GPUAllowMemoryGrowth()
pthTensorboard = join(wrkspc, 'dev', 'tensorboard')

if recompileDataset:
    # states, rewards, doneFlags, successFlags
    filePerGame = glob(join(wrkspc, 'dev', 'gameDataNewInitialsOnly', 'env3*.p'))
    print(len(filePerGame))

    # initial loop to determine maximum value of rewards to normalize
    lens, rewardsAll, i = [], [], 1
    print('Recompiling ...')
    for data in filePerGame[:maxSamples]:
        filehandler = open(data, 'rb')
        data = load(filehandler)
        filehandler.close()
        rewards = data['rewards']
        rewardsAll += rewards
        lens.append(len(rewards))
        i += 1
    # print('debug min reward', np.min(rewardsAll))
    # print('debug average reward', np.mean(rewardsAll))
    # print('debug max reward', np.max(rewardsAll))
    # print('debug max len', np.max(lens))
    # print('debug shape rewardsAll', np.shape(rewardsAll))

    X, Y, i = [], [], 1
    print('Recompiling again ...')
    for data in filePerGame[:maxSamples]:
        filehandler = open(data, 'rb')
        data = load(filehandler)
        filehandler.close()
        stresses = data['stressesNormalized']
        # states = data['statesNormalized']
        states = data['headsFullField']
        # last 4 entries: this needs to be e.g. well Q and location
        stress = np.asarray(stresses[0])
        state = np.asarray(states[0])
        state = state.flatten()

        # is this correct?

        state = np.divide(state, env.maxH)
        # state = np.divide(state-env.minH, env.maxH-env.minH)
        X.append(stress)
        Y.append(state)
        print('i', i, len(filePerGame))
        i += 1


    X = np.asarray(X)
    Y = np.asarray(Y)
    lenInput = len(X[0])
    lenOutput = len(X[0])
    # print(np.shape(X))
    # print(np.shape(Y))

    # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialXtrain.p'), 'wb')
    # dump(X, filehandler, protocol=4)
    # filehandler.close()
    # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialYtrain.p'), 'wb')
    # dump(Y, filehandler, protocol=4)
    # filehandler.close()
    np.save(join(wrkspc, 'dev', 'surrogateInitialXtrain'), X)
    np.save(join(wrkspc, 'dev', 'surrogateInitialYtrain'), Y)

print('Done recompiling')

# min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler.fit(X)
# X = min_max_scaler.transform(X)
# # X_test_scale = min_max_scaler.transform(XX_test)


def generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter = 0

  while 1:
    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    # restarting counter to yield data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0


if reTrain:

    for i in range(nRandomModels):
        learningRate = 10**(np.random.uniform(learningRateInterval[0], learningRateInterval[1]))

        if not recompileDataset:
            # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialXtrain.p'), 'rb')
            # X = load(filehandler)
            # filehandler.close()
            # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialYtrain.p'), 'rb')
            # Y = load(filehandler)
            # filehandler.close()
            X = np.load(join(wrkspc, 'dev', 'surrogateInitialXtrain.npy'))[0:maxSamples]
            Y = np.load(join(wrkspc, 'dev', 'surrogateInitialYtrain.npy'))[0:maxSamples]

        lenInput = len(X[0])
        lenOutput = len(Y[0])
        nPredictions = len(Y[0])

        print('debug type', Y[0][0].dtype)


        print('debug trainSamplesLimit', trainSamplesLimit)
        # if trainSamplesLimit != None:
        #     X = X[0:trainSamplesLimit]
        #     Y = Y[0:trainSamplesLimit]


        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Dropout

        # 250, 500, 100

        if continueModelTrain == True:
            model = load_model(join(wrkspc, 'dev', modelName + '.h5'))
        else:    
            model = Sequential()
            model.add(Dense(input_dim=lenInput, units=64))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(input_dim=64, units=256))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(input_dim=256, units=1024))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(input_dim=1024, units=5012))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(Dense(input_dim=1024, units=24576))
            # model.add(BatchNormalization())
            # model.add(Activation('relu'))
            #model.add(Dropout(0.2))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            # model.add(Dense(input_dim=256, units=256))
            # model.add(BatchNormalization())
            # model.add(Activation('sigmoid'))
            model.add(Dense(input_dim=5012, units=lenOutput))
            # model.add(BatchNormalization())
            # model.add(Activation('linear'))
            # model.add(Activation('tanh'))


        import tensorflow as tf
        print(tf.config.list_physical_devices('GPU'))

        optimizer = Adam(learning_rate=learningRate,
            beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=True)

        # model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.compile(loss='mse', optimizer=optimizer)
        # model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer)

        # tensorboard_callback = TensorBoard(log_dir=pthTensorboard)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, 
            verbose=0, mode='min')
        checkpoint = ModelCheckpoint(join(wrkspc, 'dev', 'modelCheckpoints',
            modelName) + '.h5',
            verbose=0, monitor='val_loss', save_best_only=True,
            mode='auto', restore_best_weights=True)

        # results = model.fit(X, [Y[:, i] for i in range(lenOutput)],
        #     batch_size=batch_size, epochs=epochs, validation_split=0.5, shuffle=True,
        #     callbacks=[tensorboard_callback, earlyStopping, checkpoint],
        #     verbose=1)


        print(np.shape(X))
        print(np.shape(Y))
        print('batch_size', batch_size)


        # https://stackoverflow.com/questions/58150519/resourceexhaustederror-oom-when-allocating-tensor-in-keras

        '''
        CURRENTLY FEEDING IN-SAMPLE DATA
        FIRST TRYING TO OVERFIT
        THEN ADD SEPARATE TEST DATA
        '''

        results = model.fit_generator(generator(X,Y,batch_size),
            use_multiprocessing=False,
            epochs=epochs,
            steps_per_epoch = len(Y)/batch_size,
            validation_data = generator(X,Y,batch_size*2),
            validation_steps = len(Y)/batch_size*2,
            callbacks=[earlyStopping, checkpoint],
            verbose=1)

        # results = model.fit(X, Y,
        #     use_multiprocessing=False,
        #     batch_size=batch_size, epochs=epochs, validation_split=0.5, shuffle=True,
        #     callbacks=[earlyStopping, checkpoint],
        #     verbose=1)
        # print('time taken', t0 - time.time())

        from os import mkdir
        from os.path import exists
        if not exists(join(wrkspc, 'dev', 'modelCheckpoints')):
            mkdir(join(wrkspc, 'dev', 'modelCheckpoints'))
        vallossMin = min(results.history['val_loss'])
        with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + 'History_val_loss' + f'{vallossMin:03f}' + '.p'), 'wb') as f:
            dump(results.history, f)

        t0 = time.time()
        # print(prediction)
        # print('time taken', t0 - time.time())
        # print('example sim', 60*(t0 - time.time()))

        model.save(join(wrkspc, 'dev', modelName + '.h5'))

    if not reTrain:
        model = load_model(join(wrkspc, 'dev', modelName + '.h5'))


    # going through test set
    if recompileDataset:
        # states, rewards, doneFlags, successFlags
        filePerGame = glob(join(wrkspc, 'dev', 'gameDataTest', 'env3*.p'))

        X, Y = [], []
        for data in filePerGame:
            filehandler = open(data, 'rb')
            data = load(filehandler)
            filehandler.close()
            stresses = data['stressesNormalized']
            # states = data['statesNormalized']
            states = data['headsFullField']
            # last 4 entries: this needs to be e.g. well Q and location
            stress = np.asarray(stresses[0])
            state = np.asarray(states[0])
            state = np.divide(state-env.minH, env.maxH-env.minH)
            state = state.flatten()
            X.append(stress)
            Y.append(state)

        Xtest = np.asarray(X)
        Ytest = np.asarray(Y)

        # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialXtest.p'), 'wb')
        # dump(Xtest, filehandler)
        # filehandler.close()
        # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialYtest.p'), 'wb')
        # dump(Ytest, filehandler)
        # filehandler.close()
        np.save(join(wrkspc, 'dev', 'surrogateInitialXtest'), Xtest)
        np.save(join(wrkspc, 'dev', 'surrogateInitialYtest'), Ytest)

    if not recompileDataset:
        # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialXtest.p'), 'rb')
        # Xtest = load(filehandler)
        # filehandler.close()
        # filehandler = open(join(wrkspc, 'dev', 'surrogateInitialYtest.p'), 'rb')
        # Ytest = load(filehandler)
        # filehandler.close()
        Xtest = np.load(join(wrkspc, 'dev', 'surrogateInitialXtest.npy'))[0:maxSamples]
        Ytest = np.load(join(wrkspc, 'dev', 'surrogateInitialYtest.npy'))[0:maxSamples]

    if reTest:

        if testSamplesLimit != None:
            Xtest = Xtest[0:testSamplesLimit]
            Ytest = Ytest[0:testSamplesLimit]
        lenInput = len(Xtest[0])
        lenOutput = len(Ytest[0])
        nPredictions = len(Ytest[0])

        predsAll, simsAll = [], []
        for it in range(nPredictions):
            predsAll.append([])
            simsAll.append([])

        # iterating through test samples
        for i in range(len(Xtest)):
            # actually no need to predict wellQ and wellCoords
            # t0 = time.time()
            prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
            # print(time.time() - t0)
            predictionFlatten = np.array(prediction).flatten()

            # is this right?
            # prediction = env.observationsVectorToDict(predictionFlatten)
            prediction = unnormalizeInitial(prediction, env)
            # prediction = np.array(env.observationsDictToVector(prediction))#.flatten()

            # simulated = env.observationsVectorToDict(Ytest[i, :])
            simulated = Ytest[i, :]
            simulated = unnormalizeInitial(simulated, env)
            # simulated = np.array(env.observationsDictToVector(simulated)).flatten()

            for k in range(nPredictions):
                predsAll[k].append(prediction[k])
                simsAll[k].append(simulated[k])

            if (i % 1000 == 0):
                # print('predict', i, len(Xtest), 'prediction', prediction, 'simulated', simulated)
                print('predict', i, len(Xtest))

        for i in range(nPredictions):
            plt.figure(1)
            plt.subplot(211)
            sim = simsAll[i]
            pred = predsAll[i]
            plt.scatter(sim, pred, s=0.4, lw=0., marker='.')
            plt.xlim(left=0, right=10.5)
            plt.ylim(bottom=0, top=10.5)
            plt.subplot(212)
            plt.hist(sim, bins=30)
            plt.xlim(left=0, right=10.5)
            plt.savefig(join(wrkspc, 'dev', modelName + 'pred' + str(i+1).zfill(2) + '.png'), dpi=1000)
            plt.close('all')