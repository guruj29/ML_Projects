#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Space-Object-Classification-using-classification-algorithms" data-toc-modified-id="Space-Object-Classification-using-classification-algorithms-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Space Object Classification using classification algorithms</a></span><ul class="toc-item"><li><span><a href="#Visualization" data-toc-modified-id="Visualization-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Visualization</a></span></li><li><span><a href="#Data-Pre-Processing" data-toc-modified-id="Data-Pre-Processing-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Data Pre-Processing</a></span></li></ul></li><li><span><a href="#KNN-Classifier" data-toc-modified-id="KNN-Classifier-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>KNN Classifier</a></span></li><li><span><a href="#Support-Vector-Machine" data-toc-modified-id="Support-Vector-Machine-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Support Vector Machine</a></span></li><li><span><a href="#Decision-Tree-Classifier" data-toc-modified-id="Decision-Tree-Classifier-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Decision Tree Classifier</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#Comparative-Analysis" data-toc-modified-id="Comparative-Analysis-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Comparative Analysis</a></span></li></ul></div>

# # Space Object Classification using classification algorithms

# This notebook contains machine learning models to identify space objects.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


sp_objs = pd.read_csv('tle2sv.csv')


# ## Visualization

# In[3]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[4]:


threedee = plt.figure(figsize=(50,40)).gca(projection='3d')

threedee.scatter(sp_objs['x'],sp_objs['y'],sp_objs['z'])
threedee.set_xlabel('X-position',fontsize=50)
threedee.set_ylabel('Y-position',fontsize=50)
threedee.set_zlabel('Z-position',fontsize=50)
plt.tight_layout()
plt.show()


# In[5]:


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits.mplot3d import Axes3D
cos = np.cos
sin = np.sin
pi = np.pi
dot = np.dot

fig = plt.figure(figsize=(50,40))  # Square figure
ax = fig.add_subplot(111,projection='3d')
ax.view_init(6,None)
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([10, 5, 10, 1]))


max_radius = 0
def plotEarth():
    "Draw Earth as a globe at the origin"
    Earth_radius = 6371 # km
    global max_radius
    max_radius = max(max_radius, Earth_radius)
    
    # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    coefs = (1, 1, 1)  

    # Radii corresponding to the coefficients:
    rx, ry, rz = [Earth_radius/np.sqrt(coef) for coef in coefs]

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g')

def plotOrbit(semi_major_axis, eccentricity=0, inclination=0, 
              right_ascension=0, argument_perigee=0, true_anomaly=0, label=None):
    "Draws orbit around an earth in units of kilometers."
    # Rotation matrix for inclination
    inc = inclination * pi / 180.;
    R = np.matrix([[1, 0, 0],
                   [0, cos(inc), -sin(inc)],
                   [0, sin(inc), cos(inc)]    ])

    # Rotation matrix for argument of perigee + right ascension
    rot = (right_ascension + argument_perigee) * pi/180
    R2 = np.matrix([[cos(rot), -sin(rot), 0],
                    [sin(rot), cos(rot), 0],
                    [0, 0, 1]    ])    

    ### Draw orbit
    theta = np.linspace(0,2*pi, 360)
    r = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(theta))

    xr = r*cos(theta)
    yr = r*sin(theta)
    zr = 0 * theta

    pts = np.array(list(zip(xr,yr,zr)))
    
    # Rotate by inclination
    # Rotate by ascension + perigee

    mat_mul = np.dot(R,R2)
    pts =np.dot(mat_mul,np.transpose(pts))
    pts = np.transpose(pts)

    # Turn back into 1d vectors
    xr,yr,zr = pts[:,0].A.flatten(), pts[:,1].A.flatten(), pts[:,2].A.flatten()

    # Plot the orbit
    ax.plot(xr, yr, zr, '-')
    # plt.xlabel('X (km)')
    # plt.ylabel('Y (km)')
    # plt.zlabel('Z (km)')

    # Plot the satellite
    sat_angle = true_anomaly * pi/180
    satr = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(sat_angle))
    satx = satr * cos(sat_angle)
    saty = satr * sin(sat_angle)
    satz = 0

    sat = (R * R2 * np.matrix([satx, saty, satz]).T ).flatten()
    satx = sat[0,0]
    saty = sat[0,1]
    satz = sat[0,2]

    c = np.sqrt(satx*satx + saty*saty)
    lat = np.arctan2(satz, c) * 180/pi
    lon = np.arctan2(saty, satx) * 180/pi
    #print ("%s : Lat: %g° Long: %g" % (label, lat, lon))
    
    # Draw radius vector from earth
    # ax.plot([0, satx], [0, saty], [0, satz], 'r-')
    # Draw red sphere for satellite
    ax.plot([satx],[saty],[satz], 'ro')

    global max_radius
    max_radius = max(max(r), max_radius)

    # Write satellite name next to it
    #if label:
     #   ax.text(satx, saty, satz, label, fontsize=2)
def doDraw():
    # Adjustment of the axes, so that they all have the same span:
   
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
   
    # Draw figure
    fig.tight_layout()
    plt.show()
    
import numpy as np
from datetime import datetime, timedelta
import pytz
#import graphics
import urllib
import urllib.request

pdt = pytz.timezone('US/Pacific')

sqrt = np.sqrt
pi = np.pi
sin = np.sin
cos = np.cos

# Standard Gravitational parameter in km^3 / s^2 of Earth
GM = 398600.4418

def splitElem(tle):
    "Splits a two line element set into title and it's two lines with stripped lines"
    return map(lambda x: x.strip(), tle.split('\n'))

def checkValid(tle):
    "Checks with checksum to make sure element is valid"
    title, line1, line2 =  splitElem(tle)

    return line1[0] == '1' and line2[0] == '2' and            line1[2:7] == line2[2:7] and            int(line1[-1]) == doChecksum(line1) and int(line2[-1]) == doChecksum(line2)

def stringScientificNotationToFloat(sn):
    "Specific format is 5 digits, a + or -, and 1 digit, ex: 01234-5 which is 0.01234e-5"
    return 0.00001*float(sn[5]) * 10**int(sn[6:])

def eccentricAnomalyFromMean(mean_anomaly, eccentricity, initValue,
                             maxIter=500, maxAccuracy = 0.0001):
    """Approximates Eccentric Anomaly from Mean Anomaly
       All input and outputs are in radians"""
    mean_anomaly = mean_anomaly
    e0 = initValue
    for x in range(maxIter):
        e1 = e0 - (e0 - eccentricity * sin(e0) - mean_anomaly) / (1.0 - eccentricity * cos(e0))
        if (abs(e1-e0) < maxAccuracy):
            break
    return e1

def pretty_print(tle, printInfo = True, labels = True):
    "Returns commented information on a two line element"
    title, line1, line2 =  splitElem(tle)
    if not checkValid(tle):
        print ("Invalid element.")
        return

    satellite_number                                        = int(line1[2:7])
    classification                                          = line1[7:8]
    #international_designator_year                           = int((line1[9:11]).strip())
    international_designator_launch_number                  = int(line1[11:14])
    international_designator_piece_of_launch                = line1[14:17]
    epoch_year                                              = int(line1[18:20])
    epoch                                                   = float(line1[20:32])
    first_time_derivative_of_the_mean_motion_divided_by_two = float(line1[33:43])
    second_time_derivative_of_mean_motion_divided_by_six    = stringScientificNotationToFloat(line1[44:52])
    bstar_drag_term                                         = stringScientificNotationToFloat(line1[53:61])
    the_number_0                                            = float(line1[62:63])
    element_number                                          = float(line1[64:68])
    checksum1                                               = float(line1[68:69])

    satellite        = int(line2[2:7])
    inclination      = float(line2[8:16])
    right_ascension  = float(line2[17:25])
    eccentricity     = float(line2[26:33]) * 0.0000001
    argument_perigee = float(line2[34:42])
    mean_anomaly     = float(line2[43:51])
    mean_motion      = float(line2[52:63])
    revolution       = float(line2[63:68])
    checksum2        = float(line2[68:69])

    # Inferred Epoch date
    year = 2000 + epoch_year if epoch_year < 70 else 1900 + epoch_year
    epoch_date = datetime(year=year, month=1, day=1, tzinfo=pytz.utc) + timedelta(days=epoch-1) # Have to subtract one day to get correct midnight

    # Time difference of now from epoch, offset in radians
    diff = datetime.now().replace(tzinfo=pytz.utc) + timedelta(hours=8) - epoch_date # Offset for PDT
    diff_seconds = 24*60*60*diff.days + diff.seconds + 1e-6*diff.microseconds # sec
    #print ("Time offset: %s" % diff)
    motion_per_sec = mean_motion * 2*pi / (24*60*60) # rad/sec
    #print ("Radians per second: %g" % motion_per_sec)
    offset = diff_seconds * motion_per_sec #rad
    #print ("Offset to apply: %g" % offset)
    mean_anomaly += offset * 180/pi % 360

    # Inferred period
    day_seconds = 24*60*60
    period = day_seconds * 1./mean_motion

    # Inferred semi-major axis (in km)
    semi_major_axis = ((period/(2*pi))**2 * GM)**(1./3)

    # Inferred true anomaly
    eccentric_anomaly = eccentricAnomalyFromMean(mean_anomaly * pi/180, eccentricity, mean_anomaly * pi/180)
    true_anomaly = 2*np.arctan2(sqrt(1+eccentricity) * sin(eccentric_anomaly/2.0), sqrt(1-eccentricity) * cos(eccentric_anomaly/2.0))
    # Convert to degrees
    eccentric_anomaly *= 180/pi
    true_anomaly *= 180/pi

    if (printInfo):
        print( "----------------------------------------------------------------------------------------")
        print(tle)
        print( "---")
        print("Satellite Name                                            = %s" % title)
        print( "Satellite number                                          = %g (%s)" % (satellite_number, "Unclassified" if classification == 'U' else "Classified"))
        print( "International Designator                                  = YR: %02d, LAUNCH #%d, PIECE: %s" % (international_designator_year, international_designator_launch_number, international_designator_piece_of_launch))
        print( "Epoch Date                                                = %s  (YR:%02d DAY:%.11g)" % (epoch_date.strftime("%Y-%m-%d %H:%M:%S.%f %Z"), epoch_year, epoch))
        print( "First Time Derivative of the Mean Motion divided by two   = %g" % first_time_derivative_of_the_mean_motion_divided_by_two)
        print( "Second Time Derivative of Mean Motion divided by six      = %g" % second_time_derivative_of_mean_motion_divided_by_six)
        print( "BSTAR drag term                                           = %g" % bstar_drag_term)
        print( "The number 0                                              = %g" % the_number_0)
        print( "Element number                                            = %g" % element_number)
        print( "Inclination [Degrees]                                     = %g°" % inclination)
        print( "Right Ascension of the Ascending Node [Degrees]           = %g°" % right_ascension)
        print( "Eccentricity                                              = %g" % eccentricity)
        print( "Argument of Perigee [Degrees]                             = %g°" % argument_perigee)
        print( "Mean Anomaly [Degrees] Anomaly                            = %g°" % mean_anomaly)
        print( "Eccentric Anomaly                                         = %g°" % eccentric_anomaly)
        print( "True Anomaly                                              = %g°" % true_anomaly)
        print( "Mean Motion [Revs per day] Motion                         = %g" % mean_motion)
        print( "Period                                                    = %s" % timedelta(seconds=period))
        print( "Revolution number at epoch [Revs]                         = %g" % revolution)

        print ("semi_major_axis = %gkm" % semi_major_axis)
        print ("eccentricity    = %g" % eccentricity)
        print ("inclination     = %g°" % inclination)
        print ("arg_perigee     = %g°" % argument_perigee)
        print ("right_ascension = %g°" % right_ascension)
        print ("true_anomaly    = %g°" % true_anomaly)
        print ("----------------------------------------------------------------------------------------")

    if labels:
        #print("if with labels ran , now plotOrbit must run")
        plotOrbit(semi_major_axis, eccentricity, inclination,right_ascension, argument_perigee, true_anomaly, title)
    else:
        plotOrbit(semi_major_axis, eccentricity, inclination,right_ascension, argument_perigee, true_anomaly)

def doChecksum(line):
    """The checksums for each line are calculated by adding the all numerical digits on that line, including the 
       line number. One is added to the checksum for each negative sign (-) on that line. All other non-digit 
       characters are ignored.
       @note this excludes last char for the checksum thats already there."""
    return sum(map(int, filter(lambda c: c >= '0' and c <= '9', line[:-1].replace('-','1')))) % 10



plotEarth()

f = open('1000_obj.txt')

elem = ""
counter = 0
for line in f:
    
    elem += str(line)
    if (line[0] == '2'):
        elem = elem.strip()
        #if elem.startswith("ISS"):
        pretty_print(elem, printInfo=False, labels=True)
        elem = ""

doDraw()


# ## Data Pre-Processing 

# In[6]:


from skyfield.api import Topos, load


# In[7]:


small_data = load.tle_file('15June_2020_3le.txt')
print("loaded",len(small_data),"Space Objects")


# In[8]:


from tletools import TLE
from tletools.pandas import*


# In[9]:


small_df = load_dataframe('15June_2020_3le.txt')


# In[10]:


objectType =[]
labels =[]
name = small_df['name']
name = np.asarray(name)
name_list = name.tolist()
for i in range(0,len(name_list)):
    if('R/B' in name_list[i]):
        objectType.append('R/B')
        labels.append(0)
    elif('DEB' in name_list[i]):
        objectType.append('Deb')
        labels.append(1)
    else:
        objectType.append('Satellite')
        labels.append(2)

    


# In[11]:


objectType
small_df['objtype'] = pd.DataFrame(data= objectType)


# In[12]:


df = small_df.drop(columns=['classification','int_desig','epoch_year','dn_o2','ddn_o6','bstar','set_num'])


# In[13]:


df[df['epoch_day'] == df['epoch_day'].max()]


# In[14]:


import skyfield as sf
from skyfield.api import Topos, load
from skyfield.api import EarthSatellite
import datetime
from orbit_predictor.sources import get_predictor_from_tle_lines
from sgp4.api import Satrec

ts = load.timescale(builtin=True)

t = ts.utc(2020, 6, 18, 15, 10, 26)  # max epoch in the data set 

with open('15June_2020_3le.txt', 'r') as f:
    tle_list = [line.strip() for line in f.read().split('\n')
                if line is not '']
pos_data = []
vel_data = []
for i in range(1, len(tle_list)-1, 3):  # every two lines
    temp = {}
    temp['tle1'] = tle_list[i]
    temp['tle2'] = tle_list[i+1]
    satellite    = sf.sgp4lib.EarthSatellite(tle_list[i],tle_list[i+1])
    geocentric = satellite.at(t)
    pos_data.append(geocentric.position.km)
    vel_data.append(geocentric.velocity.km_per_s)


# In[15]:


sv_data = np.hstack((pos_data, vel_data))


# In[16]:


len(sv_data)


# In[17]:


import pandas as pd
prop_sv_df = pd.DataFrame(data = sv_data, columns=['x','y','z','xdot','ydot','zdot'])


# In[18]:


prop_sv_df['Objtype'] = df['objtype']


# In[19]:


prop_sv_df['norad_Id'] = df['norad']


# In[20]:


prop_sv_df['labels'] = pd.DataFrame(data=labels)


# In[21]:


prop_sv_df.isnull().sum()


# In[22]:


prop_sv_df['labels'].unique()


# In[23]:


prop_sv_df = prop_sv_df.dropna()


# In[24]:


onlysat = prop_sv_df[prop_sv_df['Objtype']=='Satellite']
onlysat['labels'].unique()


# In[25]:


X = prop_sv_df[['x','y','z','xdot','ydot','zdot']]


# In[26]:


Y = prop_sv_df['labels']


# In[27]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[28]:


xtrain.shape


# In[29]:


ytrain.shape


# In[30]:


ytest.shape


# In[31]:


ytrain.shape


# # KNN Classifier

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors=14)    # create object for that algorithm 


# In[33]:


kmodel.fit(xtrain,ytrain)


# In[34]:


Yp_train = kmodel.predict(xtrain)
(Yp_train == ytrain).sum()/len(xtrain)


# In[35]:


# testing accuracy or validation accuracy
Yp_test = kmodel.predict(xtest)
(Yp_test == ytest).sum()/len(xtest)


# In[36]:


trn_acc = []
tstg_acc = []
for i in range(1,15):
    kmodel = KNeighborsClassifier(n_neighbors=i)

    kmodel.fit(xtrain,ytrain)

    ytrain_p = kmodel.predict(xtrain)
    trn_acc.append((ytrain_p == ytrain).sum()/len(xtrain))

    ytest_p = kmodel.predict(xtest)
    tstg_acc.append((ytest_p == ytest).sum()/len(xtest))


# In[37]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of nieghbours")
plt.ylabel("Accuray of the model")
plt.plot(range(1,15), trn_acc, label='training accuracy')
plt.plot(range(1,15), tstg_acc, label='testing accuracy')
plt.legend()
plt.show()


# # Support Vector Machine

# In[38]:


#Scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sc = StandardScaler()
sc.fit(xtrain)
X_train_std = sc.transform(xtrain)
X_test_std = sc.transform(xtest)


# In[39]:


#Applying SVC (Support Vector Classification)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=5, gamma=.10, C=1.0)
svm.fit(X_train_std, ytrain)
print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train_std, ytrain)))
print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test_std, ytest)))


# # Decision Tree Classifier

# In[52]:


from sklearn import tree

#Create tree object
decision_tree = tree.DecisionTreeClassifier(criterion='gini',max_depth=15)

#Train DT based on scaled training set
decision_tree.fit(X_train_std, ytrain)

#Print performance
print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, ytrain)))
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, ytest)))


# In[41]:


trn_acc = []
tstg_acc = []
for i in range(1,20):
    decision_tree = tree.DecisionTreeClassifier(criterion='gini',max_depth=i,random_state=42)

    decision_tree.fit(X_train_std, ytrain)

    trn_acc.append(decision_tree.score(X_train_std, ytrain))

    tstg_acc.append(decision_tree.score(X_test_std, ytest))


# In[42]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Max Depth")
plt.ylabel("Score of the model")
plt.plot(range(1,20), trn_acc, label='training accuracy')
plt.plot(range(1,20), tstg_acc, label='testing accuracy')
plt.legend()
plt.show()


# # Random Forest

# In[53]:


#Applying RandomForest
from sklearn.ensemble import RandomForestClassifier

#Create Random Forest object
random_forest = RandomForestClassifier(n_estimators=8)

#Train model
random_forest.fit(X_train_std, ytrain)

#Print performance
print('The accuracy of the Random Forest classifier on training data is {:.2f}'.format(random_forest.score(X_train_std, ytrain)))
print('The accuracy of the Random Forest classifier on test data is {:.2f}'.format(random_forest.score(X_test_std, ytest)))


# In[44]:


trn_accrf = []
tstg_accrf = []
nest = [10,20,30,40,50,60,70,80,90,100]
for i in nest:
    random_forest = RandomForestClassifier(n_estimators=i)

    random_forest.fit(X_train_std, ytrain)

    trn_accrf.append(random_forest.score(X_train_std, ytrain))

    tstg_accrf.append(random_forest.score(X_test_std, ytest))


# In[45]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("No. of Estimators")
plt.ylabel("Score of the model")
plt.plot(range(1,11), trn_accrf, label='training accuracy')
plt.plot(range(1,11), tstg_accrf, label='testing accuracy')
plt.legend()
plt.show()


# In[46]:


trn_accrf = []
tstg_accrf = []
for i in range(1,11):
    random_forest = RandomForestClassifier(n_estimators=i)

    random_forest.fit(X_train_std, ytrain)

    trn_accrf.append(random_forest.score(X_train_std, ytrain))

    tstg_accrf.append(random_forest.score(X_test_std, ytest))


# In[47]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("No. of Estimators")
plt.ylabel("Score of the model")
plt.plot(range(1,11), trn_accrf, label='training accuracy')
plt.plot(range(1,11), tstg_accrf, label='testing accuracy')
plt.legend()
plt.show()


# In[51]:


Yc = Y.map({0:'r', 1:'g',2:'b'})
labels_plot = ['R/B','DEB','Satellite']
threedee1 = plt.figure(figsize=(50,40)).gca(projection='3d')
threedee1.scatter(prop_sv_df['x'],prop_sv_df['y'],prop_sv_df['z'],c=Yc)
threedee1.set_xlabel('X-position',fontsize=50)
threedee1.set_ylabel('Y-position',fontsize=50)
threedee1.set_zlabel('Z-position',fontsize=50)
plt.legend(labels= labels_plot)
plt.tight_layout()
plt.show()


# # Comparative Analysis

# In[54]:


labels = ["KNeighborsClassifier", "SupportVectorMachines", "DecisionTreeClassifier", "RandomForestClassifier"]
training_accuracy = [0.68 , 0.66 , 0.78 , 0.97 ]
testing_accuracy = [0.65 ,  0.64 , 0.68 , 0.67 ]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig,ax = plt.subplots(figsize = (8 , 8))
rects1 = ax.bar(labels, training_accuracy, label='training')
rects2 = ax.bar(labels , testing_accuracy, label='testing')


# Text and title for labels.
ax.set_ylabel('score')
ax.set_title('Score of different Algorithms')
ax.set_xticks(x)
ax.set_xticklabels((labels) , rotation = 30)
ax.legend()

fig.tight_layout()
plt.savefig('Score_of_different_Algorithms.png')

plt.show()


# In[57]:


training_accuracy = [0.68 , 0.66 , 0.78 , 0.97 ]
testing_accuracy = [0.65 ,  0.64 , 0.68 , 0.67 ]
labels = ["KNeighborsClassifier", "SupportVectorMachines", "DecisionTreeClassifier", "RandomForestClassifier"]

plt.title('Accuracy')

plt.plot( training_accuracy, label='training accuracy')
plt.plot(testing_accuracy, label='testing accuracy' )
plt.legend()
fig.tight_layout()
plt.savefig('Accuracy.png')
plt.show()


# In[ ]:




