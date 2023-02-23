import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def  EnhancedAlignmentTerm(align_Matrix):
    return  ((align_Matrix + 1)**2)/4

# Alignment Term
def AlignmentTerm(dFM,dGT):

    #compute global mean
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)

    #compute the bias matrix
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT

    #compute alignment matrix
    align_Matrix = 2*(align_GT*align_FM)/(align_GT*align_GT + align_FM*align_FM + 0.00001)

    # Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2)

    return align_Matrix

def Emeasure(FM,GT):

    dFM = FM
    dGT = GT

    #Special case:
    if (np.sum(dGT)==0):# if the GT is completely black
        enhanced_matrix = 1.0 - dFM #only calculate the black area of intersection
    elif(np.mean(dGT)==1): #if the GT is completely white
        enhanced_matrix = dFM # %only calcualte the white area of intersection
    else:
        #Normal case:
        #1.compute alignment matrix
        align_matrix = AlignmentTerm(dFM,dGT)
        #print(align_matrix)

        #2.compute enhanced alignment matrix
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
        
    #3.Emeasure score
    w,h = GT.shape[0],GT.shape[1]
    score = np.sum(enhanced_matrix)/(w*h - 1 + 0.000001)
    return score


def roc(im_gt, smap, stride):
    if stride ==1:
        tp,tn,fp,fn, E_score = np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float)
        for p in range(0,255):
            _,BW = cv2.threshold(smap,p,255,cv2.THRESH_BINARY)
            BW = BW/255.0
            tp_temp= BW*im_gt
            tp[p] = np.sum(tp_temp)
            fn_temp = (1-BW)*im_gt
            fn[p] = np.sum(fn_temp)
            fp_temp = BW*(1-im_gt)
            fp[p] = np.sum(fp_temp)
            tn_temp = (1-BW)*(1-im_gt)
            tn[p] = np.sum(tn_temp)
            E_score[p] = Emeasure(BW,im_gt)
            #print(E_score[p])

    else:
        tp,tn,fp,fn,E_score=0.0,0.0,0.0,0.0,0.0
        _,BW = cv2.threshold(smap,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        BW = BW/255.0
        tp_temp= BW*im_gt
        tp = np.sum(tp_temp)
        fn_temp = (1-BW)*im_gt
        fn = np.sum(fn_temp)
        fp_temp = BW*(1-im_gt)
        fp = np.sum(fp_temp)
        tn_temp = (1-BW)*(1-im_gt)
        tn = np.sum(tn_temp)
        E_score = Emeasure(BW,im_gt)

    return tp,tn,fp,fn,E_score


def centroid(GT):
    rows,cols = GT.shape[0],GT.shape[1]

    if(np.sum(GT)==0):
        X = round(cols/2)
        Y = round(rows/2)
    else:     
        total=np.sum(GT)
        i=np.array(range(0,rows))
        j=np.array(range(0,cols))
        X=round(np.sum(np.sum(GT,0)*i)/total)
        Y=round(np.sum(np.sum(GT,1)*j)/total)
        
    return X,Y

def divideGT(GT,X,Y):
    # % LT - left top;
    # % RT - right top;
    # % LB - left bottom;
    # % RB - right bottom;

    #width and height of the GT
    hei,wid = GT.shape[0],GT.shape[1]
    area = wid * hei

    #copy the 4 regions 
    LT = GT[0:Y,0:X]
    RT = GT[0:Y,X:wid]
    LB = GT[Y:hei,0:X]
    RB = GT[Y:hei,X:wid]

    #The different weight (each block proportional to the GT foreground region).
    w1 = (X*Y)/area
    w2 = ((wid-X)*Y)/area
    w3 = (X*(hei-Y))/area
    w4 = 1.0 - w1 - w2 - w3
    return LT,RT,LB,RB,w1,w2,w3,w4

#divide the GT into 4 regions according to the centroid of the GT and return the weights
#Divide the prediction into 4 regions according to the centroid of the GT 
def Divideprediction(prediction,X,Y):
    #width and height of the prediction
    hei,wid = prediction.shape[0],prediction.shape[1]

    #copy the 4 regions 
    LT = prediction[0:Y,0:X]
    RT = prediction[0:Y,X:wid]
    LB = prediction[Y:hei,0:X]
    RB = prediction[Y:hei,X:wid]

    return LT,RT,LB,RB


def ssim(prediction,GT):

    dGT = GT
    hei,wid = prediction.shape[0],prediction.shape[1]
    N = wid*hei

    #Compute the mean of SM,GT
    x = np.mean(prediction)
    y = np.mean(dGT)

    #Compute the variance of SM,GT
    sigma_x2 = np.sum(np.sum((prediction - x)**2))/(N - 1 + 0.00001) #sigma_x2 = var(prediction(:))
    sigma_y2 = np.sum(np.sum((dGT - y)**2))/(N - 1 + 0.0001) #sigma_y2 = var(dGT(:));      

    #Compute the covariance between SM and GT
    sigma_xy = np.sum(np.sum((prediction - x)*(dGT - y)))/(N - 1 + 0.00001)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2)*(sigma_x2 + sigma_y2)

    if(alpha != 0):
        Q = alpha/(beta + 0.0001)
    elif(alpha == 0 and beta == 0):
        Q = 1.0
    else:
        Q = 0
    return Q

def S_region(prediction,GT):

    # find the centroid of the GT
    X,Y = centroid(GT)

    # divide GT into 4 regions
    GT_1,GT_2,GT_3,GT_4,w1,w2,w3,w4 = divideGT(GT,X,Y)

    #Divede prediction into 4 regions
    prediction_1,prediction_2,prediction_3,prediction_4 = Divideprediction(prediction,X,Y)

    #Compute the ssim score for each regions
    Q1 = ssim(prediction_1,GT_1)
    Q2 = ssim(prediction_2,GT_2)
    Q3 = ssim(prediction_3,GT_3)
    Q4 = ssim(prediction_4,GT_4)

    #Sum the 4 scores
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q

def Object(prediction,GT):

    #compute the mean of the foreground or background in prediction
    index = np.where(GT>0)
    x = np.mean(prediction[index])
    #compute the standard deviations of the foreground or background in prediction
    sigma_x = np.std(prediction[index])
    score = 2.0 * x/(x**2 + 1.0 + sigma_x + 0.0001)
    return score

def S_object(prediction,GT):
    # compute the similarity of the foreground in the object level
    prediction_fg = prediction
    index = np.where(GT<1)
    prediction_fg[index]=0
    O_FG = Object(prediction_fg,GT)

    #compute the similarity of the background
    prediction_bg = 1.0 - prediction
    index = np.where(GT>0)
    prediction_bg[index] = 0
    O_BG = Object(prediction_bg,np.abs(GT-1))

    # combine the foreground measure and background measure together
    u = np.mean(GT)
    Q = u * O_FG + (1 - u) * O_BG
    return Q

def StructureMeasure(prediction,GT):

    y = np.mean(GT)

    if (y==0):# if the GT is completely black
        x = np.mean(prediction)
        Q = 1.0 - x#only calculate the area of intersection
    elif(y==1):#if the GT is completely white
        x = np.mean(prediction)
        Q = x #only calcualte the area of intersection
    else:
        alpha = 0.5
        Q = alpha*S_object(prediction,GT)+(1-alpha)*S_region(prediction,GT)
        if (Q<0):
            Q=0
    return Q




def plot_PR_curve(p,r, L, colors,lines):

    N = p.shape[0]
    prop=[]
    for i in range(0,N):
        print(i)
        if i <=8:
            prop.append([colors[i],lines[0]])
        else: 
            prop.append([colors[i%8],lines[1]])

    for i in range(0,N):
        plt.plot(r[i,0:255],p[i,0:255],color=prop[i][0],linestyle=prop[i][1],label=L[i])
    plt.axis([0, 1, 0, 1])
    splt.legend()
    #print(x,data['tp'].shape)
    plt.show()

def plot_E_curve(scores, L, colors,lines):

    x = (np.array(list(range(0,scores.shape[1])))+1)
    N = scores.shape[0]
    prop=[]
    for i in range(0,N):
        print(i)
        if i <=8:
            prop.append([colors[i],lines[0]])
        else: 
            prop.append([colors[i%8],lines[1]])

    for i in range(N):
        plt.plot(x,scores[i,:],color=prop[i][0],linestyle=prop[i][1],label=L[i])


    plt.axis([0, 255, 0, 1])
    # plt.xlabel('Dropout probability p')
    # plt.ylabel('F-measure')
    plt.legend()
    #print(x,data['tp'].shape)
    plt.show()

if __name__ == '__main__':

    dataset = 'ACT'
    gt_path = 'E:\Jue\Project\Data\ACT_75mm/test\seg\gt/'
    #re_path = os.path.join('E:\Jue\MethodEvaluation\comparitivemthod-new-new',dataset)
    re_path = 'E:\Jue\RSE2023\Results'
    folders = ['GradCAM/']
    legend = ['mask']
    #f = open('E:\Jue\RSE2023\Code/'+'method.txt','a')
    #f.write('method, sf, precision, recall, ac, iou, mae, S')
    #f.write('\n')
    precision,recall,f,ac,iou, mae, S, Escore = [],[],[],[],[],[],[],[]
    precision_curve,recall_curve,f_curve, E_curve = np.zeros((len(folders),255)),np.zeros((len(folders),255)),np.zeros((len(folders),255)),np.zeros((len(folders),255))
    dis = np.zeros((256,256,3))
    for i, folder in enumerate(folders):
        results_path = os.path.join(re_path,folder,'test')
        files = os.listdir(results_path)
       

        tp,tn,fp,fn, = np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float), np.arange(0.0,255.0,1,float)
        tp_o,tn_o,fp_o,fn_o,mae_t,E_score_o = 0.0,0.0,0.0,0.0,0.0,0.0
        s_t,E_score_t = [],[]

        for j, img in enumerate(files):
            smap = cv2.imread(os.path.join(results_path,img),0)
            print(os.path.join(results_path,img))
            smap_1 = smap/255.0
            _, prediction = cv2.threshold(smap,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            prediction = prediction/255.0
            im_gt = cv2.imread(os.path.join(gt_path,img),0)
            im_gt = cv2.resize(im_gt, (256,256), interpolation=cv2.INTER_LINEAR)/255.0
            dis[:,:,2] = im_gt*255
            dis[:,:,1]= prediction*255
            cv2.imwrite(os.path.join(re_path,folder,'dis',img),dis)
        

            mae_t += np.mean(np.abs(im_gt-smap_1))
            #Changing the threshold to generate curves
            tp_1,tn_1,fp_1,fn_1,E_score_1 = roc(im_gt, smap, stride=1)
            #print(tp_1.dtype,tp.dtype)
            tp += tp_1
            tn += tn_1
            fp += fp_1
            fn += fn_1
            #print(E_score_1)
            E_score_t.append(E_score_1)

            #Using otsu to produce BW
            tp_2,tn_2,fp_2,fn_2,E_score_2 = roc(im_gt, smap, stride=0)
            tp_o += tp_2
            tn_o += tn_2
            fp_o += fp_2
            fn_o += fn_2
            E_score_o += E_score_2

            s_t.append(StructureMeasure(prediction,im_gt))

            # if j>200:
            #     break

        # precision_curve[i,:] = tp/(tp + fp)
        # recall_curve[i,:] = tp/(tp + fn)
        # f_curve[i,:] = 2*precision_curve[i,:]*recall_curve[i,:]/(precision_curve[i,:] + recall_curve[i,:])
        
        precision.append(tp_o/(tp_o + fp_o))
        recall.append(tp_o/(tp_o + fn_o))
        f.append(2*precision[i]*recall[i]/(precision[i] + recall[i]))
        ac.append((tp_o + tn_o)/(tp_o+tn_o+fp_o+fn_o))
        iou.append(tp_o/(tp_o + fp_o + fn_o))
        mae.append(mae_t/len(files))
        S.append(np.mean(np.array(s_t)))
        Escore.append(np.mean(E_score_o))
        #E_curve[i] = np.mean(E_score_t,0)

        # f.write(folder)
        # f.write('{:}{:}{:}{:}{:}{:}{:}')

    print(f,precision,recall,ac,iou,mae,S)
    #f.close()
    colors=['b','c','g','r','m','y','k','c']
    lines=['-','--']
    #print(np.array(precision_curve),np.array(recall_curve),np.array(E_curve))
    #plot_PR_curve(np.array(precision_curve),np.array(recall_curve), legend, colors, lines)
    #plot_E_curve(np.array(E_curve), legend, colors, lines)
