import numpy as np
import math
from tqdm import tqdm 
import cv2
import pickle
import sklearn.metrics as metrics
import random 
import matplotlib.pyplot as plt



class Convolution:
    def __init__(self, o_channel, f_dimension, stride, padding,alpha):
        self.o_channel = o_channel
        self.f_dimension = f_dimension
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.bias = None
        self.initialized = False
        self.i_channel = -1
        self.alpha=alpha

    def forward(self, input_list):
        self.x=input_list
        if self.initialized == False:
            self.i_channel = input_list[0].shape[0]
            self.filters = np.random.randn(self.o_channel, self.i_channel, self.f_dimension, self.f_dimension) * math.sqrt(2 / (self.f_dimension * self.f_dimension * self.i_channel))
            self.bias = np.zeros(self.o_channel)
            self.initialized = True
        # else :
        #     assert self.i_channel == input_list[0].shape[0]

        out_h = (input_list.shape[2] - self.f_dimension + 2 * self.padding) // self.stride + 1
        out_w = (input_list.shape[3] - self.f_dimension + 2 * self.padding) // self.stride + 1

        padded_input = input_list
        if self.padding != 0:
            padded_input = np.pad(input_list, ((0,), (0,), (self.padding,), (self.padding,)), 'constant', constant_values=(0.,))
        batch_str, channel_str, kern_h_str, kern_w_str = padded_input.strides

        windows = np.lib.stride_tricks.as_strided(
            padded_input,
            (input_list.shape[0], input_list.shape[1], out_h, out_w, self.f_dimension, self.f_dimension),
            (batch_str, channel_str, self.stride * kern_h_str, self.stride * kern_w_str, kern_h_str, kern_w_str)
        )

        out = np.einsum('bihwkl,oikl->bohw', windows, self.filters)

        out += self.bias[None, :, None, None]

        self.x = input_list
        self.windows=windows
        return out
    def backward(self,d_z):
        x, windows = self.x, self.windows

        if self.padding == 0:
            padding = self.f_dimension - 1
        else:
            padding = self.padding
        
        d_z_modified = d_z 
        if self.stride - 1 != 0:
            d_z_modified = np.insert(d_z_modified, range(1, d_z_modified.shape[2]), 0, axis=2)
            d_z_modified = np.insert(d_z_modified, range(1, d_z_modified.shape[3]), 0, axis=3)

        if padding>0:
            d_z_modified = np.pad(d_z_modified, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=(0.,))
        batch_str, channel_str, kern_h_str, kern_w_str = d_z_modified.strides
        dout_windows = np.lib.stride_tricks.as_strided(
            d_z_modified,
            (d_z.shape[0], d_z.shape[1], x.shape[2], x.shape[3], self.f_dimension, self.f_dimension),
            (batch_str, channel_str,  kern_h_str, kern_w_str, kern_h_str, kern_w_str)
        )
        rot_kern = np.rot90(self.filters, 2, axes=(2, 3))

        db = np.sum(d_z, axis=(0, 2, 3))
        db/=x.shape[0]

        dw = np.einsum('bihwkl,bohw->oikl', windows, d_z)
        dw/=x.shape[0]

        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)
        
        self.filters -= self.alpha * dw
        self.bias -= self.alpha * db
        return dx


class MaxPooling:
    def __init__(self, f_dimension, stride):
        self.f_dimension = f_dimension
        self.stride = stride

    def forward(self, input_list):
        output_list = []
        self.x=input_list

        out_h = int(math.floor((input_list.shape[2] - self.f_dimension) / self.stride)) + 1
        out_w = int(math.floor((input_list.shape[3] - self.f_dimension) / self.stride)) + 1

        batch_str, channel_str, kern_h_str, kern_w_str = input_list.strides
        windows = np.lib.stride_tricks.as_strided(
            input_list,
            (input_list.shape[0], input_list.shape[1], out_h, out_w, self.f_dimension, self.f_dimension),
            (batch_str, channel_str, self.stride * kern_h_str, self.stride * kern_w_str, kern_h_str, kern_w_str)
        )
        output_list = np.max(windows, axis=(4, 5))
        self.output_list=output_list
        return output_list
    
    def backward(self, d_z):
        d_x=np.zeros(self.x.shape)
        for i in range(0,self.f_dimension):
            for j in range(0,self.f_dimension):
                sliced=self.x[:, :, i:i+d_z.shape[2]*self.stride:self.stride, j:j+d_z.shape[3]*self.stride:self.stride]
                # print('sliced shape: ',sliced.shape)
                mask=(sliced == self.output_list)
                d_x[:, :, i:i+d_z.shape[2]*self.stride:self.stride, j:j+d_z.shape[3]*self.stride:self.stride]+=mask*d_z
        return d_x


class ReLU:
    def forward(self, input_list):
        self.x=input_list
        output_list=np.maximum(0,input_list)
        return output_list

    def backward(self, d_z):
        d_x=d_z*(self.x>0)
        return d_x


class Flattening:
    def forward(self, input_list):
        self.shape=input_list.shape
        output_list=input_list.ravel()
        output_list=output_list.reshape((input_list.shape[0],input_list.shape[1]*input_list.shape[2]*input_list.shape[3]))      
        return output_list
    def backward(self, d_z):
        d_x=d_z.reshape(self.shape)
        return d_x


class Dense:
    def __init__(self, o_channel,alpha):
        self.o_channel = o_channel 
        self.initialized = False
        self.alpha=alpha

    def forward(self, input_list):
        if self.initialized == False:
            self.i_channel = input_list[0].shape[0]
            self.weights = np.random.randn(self.o_channel, self.i_channel) * math.sqrt(2 / (self.o_channel))
            self.bias = np.zeros(self.o_channel)
            self.initialized = True
        # else :
        #     assert self.i_channel == input_list[0].shape[0]
        self.input=input_list
        output_list=input_list.dot(self.weights.T) + self.bias
        return output_list

    def backward(self, d_z):
        d_w=d_z.T.dot(self.input)
        d_b=np.sum(d_z,axis=0)

        d_w/=self.input.shape[0]
        d_b/=self.input.shape[0]

        self.weights-=self.alpha*d_w
        self.bias-=self.alpha*d_b
        d_x=self.weights.T.dot(d_z.T).T
        return d_x


class SoftMax:

    def forward(self, input_list):
        self.x=input_list
        mx=np.max(input_list,axis=1)
        mx=mx.reshape((input_list.shape[0],1))
        input_list=input_list-mx 
        output_list=np.exp(input_list)
        sum=np.sum(output_list,axis=1)
        sum=sum.reshape((input_list.shape[0],1))
        output_list=output_list/sum
        return output_list

    def backward(self, input_list, output_list):
        return input_list-output_list


def data_handler(which):
        # sys.stdout = open('C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\output.txt', 'w')
        # np.set_printoptions(threshold=sys.maxsize)

        all_images=[]
        all_digits=[]

        csv_file='C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\NumtaDB_with_aug\\'+which+'.csv'
        image_folder='C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\NumtaDB_with_aug\\'+which+'\\'
        # idx=1000
        with open(csv_file) as f:
            next(f)
            lines=f.readlines()
            for line in tqdm(lines):
                # if idx==0:
                #     break
                # idx-=1
                line = line.split(',')
                filename = line[0]
                digit=line[3]
                img = cv2.imread(image_folder+filename,0)
                img = cv2.resize(img,(28,28))

                img = np.array(img)
                img = 255-img
                img = img/255

                #dilate
                kernel = np.ones((5,5),np.uint8)
                img = cv2.dilate(img,kernel,iterations = 1)

                temp=np.zeros((1,28,28))
                temp[0,:,:]=img

                all_images.append(temp)
                y_correct=np.zeros(10)
                y_correct[int(digit)]=1
                all_digits.append(y_correct)
        all_images=np.array(all_images)
        all_digits=np.array(all_digits)
        return all_images,all_digits


class Lenet:
    def __init__(self,alpha):
        self.layers=[]
        self.alpha=alpha

        self.Convolution1=Convolution(6,5,1,2,alpha)
        self.layers.append(self.Convolution1)

        self.ReLU1=ReLU()
        self.layers.append(self.ReLU1)

        self.MaxPooling1=MaxPooling(2,2)
        self.layers.append(self.MaxPooling1)
        
        self.Convolution2=Convolution(16,5,1,2,alpha)
        self.layers.append(self.Convolution2)

        self.ReLU2=ReLU()
        self.layers.append(self.ReLU2)

        self.MaxPooling2=MaxPooling(2,2)
        self.layers.append(self.MaxPooling2)

        self.Flattening=Flattening()
        self.layers.append(self.Flattening)

        self.Dense1=Dense(120,alpha)
        self.layers.append(self.Dense1)

        self.Dense2=Dense(84,alpha)
        self.layers.append(self.Dense2)

        self.Dense3=Dense(10,alpha)
        self.layers.append(self.Dense3)

        self.SoftMax=SoftMax()

    def train(self,a,b):
        
        batch_size=16
        for i in tqdm(range(0,len(a),batch_size)):
            input_list=a[i:min(i+batch_size,len(a))]
            b_list=b[i:min(i+batch_size,len(a))]

            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            output_list=self.SoftMax.backward(output_list,b_list)
            for l in range(len(self.layers)-1,-1,-1):
                output_list=self.layers[l].backward(output_list)

    def validate(self,a,b):
        tot=0
        batch_size=16
        cnt=np.zeros(10)
        y_preds=[]
        y_trues=[]
        probabilities=[]
        encodings=[]

        for i in tqdm(range(0,len(a),batch_size)):
            # print(i)
            input_list=a[i:min(i+16,len(a))]
            b_list=b[i:min(i+16,len(a))]
            
            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            
            y_pred = np.argmax(output_list, axis=1)
            y_true = np.argmax(b_list, axis=1)
            
            correct=np.sum(y_pred==y_true)
            tot+=correct
            for x in y_pred:
                cnt[x]+=1

            for x in output_list:
                probabilities.append(x)
            for x in y_pred:
                y_preds.append(x)
                
            for x in y_true:
                y_trues.append(x)
                encoding=np.zeros(10)
                encoding[x]=1
                encodings.append(encoding)
            # print(i,correct)
            # print(i,y_pred,y_true)
        print(cnt)
        probabilities=np.array(probabilities)
        y_preds=np.array(y_preds)
        y_trues=np.array(y_trues)
        encodings=np.array(encodings)
        return probabilities,y_preds,y_trues,encodings

    def test(self,a):
        batch_size=16
        y_preds=[]

        for i in tqdm(range(0,len(a),batch_size)):
            # print(i)
            input_list=a[i:min(i+16,len(a))]
            
            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            
            y_pred = np.argmax(output_list, axis=1)
            
            for x in y_pred:
                y_preds.append(x)

        y_preds=np.array(y_preds)
        return y_preds
    
class Lenet_modified:
    def __init__(self,alpha):
        self.layers=[]
        self.alpha=alpha

        self.Convolution1=Convolution(6,5,1,2,alpha)
        self.layers.append(self.Convolution1)

        self.ReLU1=ReLU()
        self.layers.append(self.ReLU1)

        self.MaxPooling1=MaxPooling(2,2)
        self.layers.append(self.MaxPooling1)
        
        self.Convolution2=Convolution(16,5,1,2,alpha)
        self.layers.append(self.Convolution2)

        self.ReLU2=ReLU()
        self.layers.append(self.ReLU2)

        self.MaxPooling2=MaxPooling(2,2)
        self.layers.append(self.MaxPooling2)

        self.Convolution3=Convolution(26,5,1,2,alpha)
        self.layers.append(self.Convolution3)

        self.ReLU3=ReLU()
        self.layers.append(self.ReLU3)

        self.MaxPooling3=MaxPooling(2,2)
        self.layers.append(self.MaxPooling3)

        self.Flattening=Flattening()
        self.layers.append(self.Flattening)

        self.Dense0=Dense(220,alpha)
        self.layers.append(self.Dense0)

        self.Dense1=Dense(120,alpha)
        self.layers.append(self.Dense1)

        self.Dense2=Dense(84,alpha)
        self.layers.append(self.Dense2)

        self.Dense3=Dense(10,alpha)
        self.layers.append(self.Dense3)

        self.SoftMax=SoftMax()

    def train(self,a,b):
        
        batch_size=16
        for i in tqdm(range(0,len(a),batch_size)):
            input_list=a[i:min(i+batch_size,len(a))]
            b_list=b[i:min(i+batch_size,len(a))]

            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            output_list=self.SoftMax.backward(output_list,b_list)
            for l in range(len(self.layers)-1,-1,-1):
                output_list=self.layers[l].backward(output_list)

    def validate(self,a,b):
        tot=0
        batch_size=16
        cnt=np.zeros(10)
        y_preds=[]
        y_trues=[]
        probabilities=[]
        encodings=[]

        for i in tqdm(range(0,len(a),batch_size)):
            # print(i)
            input_list=a[i:min(i+16,len(a))]
            b_list=b[i:min(i+16,len(a))]
            
            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            
            y_pred = np.argmax(output_list, axis=1)
            y_true = np.argmax(b_list, axis=1)
            
            correct=np.sum(y_pred==y_true)
            tot+=correct
            for x in y_pred:
                cnt[x]+=1

            for x in output_list:
                probabilities.append(x)
            for x in y_pred:
                y_preds.append(x)
                
            for x in y_true:
                y_trues.append(x)
                encoding=np.zeros(10)
                encoding[x]=1
                encodings.append(encoding)
            # print(i,correct)
            # print(i,y_pred,y_true)
        print(cnt)
        probabilities=np.array(probabilities)
        y_preds=np.array(y_preds)
        y_trues=np.array(y_trues)
        encodings=np.array(encodings)
        return probabilities,y_preds,y_trues,encodings

    def test(self,a):
        batch_size=16
        y_preds=[]

        for i in tqdm(range(0,len(a),batch_size)):
            # print(i)
            input_list=a[i:min(i+16,len(a))]
            
            output_list=input_list

            for l in range(len(self.layers)):
                output_list=self.layers[l].forward(output_list)
            output_list=self.SoftMax.forward(output_list)
            
            y_pred = np.argmax(output_list, axis=1)
            
            
            for x in y_pred:
                y_preds.append(x)

        y_preds=np.array(y_preds)
        return y_preds

if __name__ == '__main__':
    alpha=0.0005
    model=Lenet(alpha)
    

    a1,b1=data_handler('training-a')
    a2,b2=data_handler('training-b')
    a3,b3=data_handler('training-c')


    a1=np.concatenate((a1,a2,a3),axis=0)
    b1=np.concatenate((b1,b2,b3),axis=0)

    c = list(zip(a1, b1))
    random.shuffle(c)
    a1, b1 = zip(*c)

    # split data into training set and validation set
    total_data=len(a1)
    training_data=0.8*total_data
    validation_data=0.2*total_data

    a1_training=np.array(a1[0:int(training_data)])
    b1_training=np.array(b1[0:int(training_data)])

    a1_validation=np.array(a1[int(training_data):])
    b1_validation=np.array(b1[int(training_data):])

    print(total_data)

    epochs=[]
    training_losses=[]
    validation_losses=[]
    validation_accuracys=[]
    validation_f1s=[]



    for epoch in range(1,26):
        print('Epoch: ',epoch)
        
        print('Training')
        model.train(a1_training,b1_training)

        print('Validating on training set')
        probabilities,y_preds,y_trues,encodings=model.validate(a1_training,b1_training)

        training_loss=metrics.log_loss(encodings,probabilities)
        print('Training Log Loss: ',training_loss)
        training_losses.append(training_loss)

        print('Validating on validation set')
        probabilities,y_preds,y_trues,encodings=model.validate(a1_validation,b1_validation)
        
        validation_accuracy=metrics.accuracy_score(y_trues,y_preds)
        print('Validation Accuracy: ',validation_accuracy)
        validation_accuracys.append(validation_accuracy)

        f1_score = metrics.f1_score(y_trues,y_preds,average='macro')
        print('Validation F1: ',metrics.f1_score(y_trues,y_preds,average='macro'))
        validation_f1s.append(f1_score)

        validation_loss=metrics.log_loss(encodings,probabilities)
        print('Validation Log Loss: ',metrics.log_loss(encodings,probabilities))
        validation_losses.append(validation_loss)

        print('Validation Confusion Matrix: ',metrics.confusion_matrix(y_trues,y_preds))
        epochs.append(epoch)

        pickle_name = 'MODEL_' + '_' + str(alpha) + '_' + str(epoch) + '.pkl'
        with open(pickle_name, 'wb') as f:
            pickle.dump(model, f)
    epochs=np.array(epochs)
    training_losses=np.array(training_losses)
    validation_losses=np.array(validation_losses)
    validation_accuracys=np.array(validation_accuracys)
    validation_f1s=np.array(validation_f1s)

    # plot training loss vs epochs
    plt.plot(epochs,training_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs')
    filename = 'C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\Training Loss_vs_Epochs_' + '_' + str(alpha) + '.png'
    plt.savefig(filename)
    plt.clf()

    # plot validation loss vs epochs
    plt.plot(epochs,validation_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epochs')
    filename = 'C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\Validation Loss_vs_Epochs_'  + '_' + str(alpha) + '.png'
    plt.savefig(filename)
    plt.clf()

    # plot validation accuracy vs epochs
    plt.plot(epochs,validation_accuracys)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    filename = 'C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\Validation Accuracy_vs_Epochs_' + '_' + str(alpha) + '.png'
    plt.savefig(filename) 
    plt.clf()

    # plot validation f1 vs epochs
    plt.plot(epochs,validation_f1s)
    plt.xlabel('Epochs')
    plt.ylabel('Validation F1')
    plt.title('Validation F1 vs Epochs')
    filename = 'C:\\Users\\iftek\\Desktop\\4-2\\472\\Offline4\\Validation F1_vs_Epochs_' + '_' + str(alpha) + '.png'
    plt.savefig(filename)
    plt.clf()