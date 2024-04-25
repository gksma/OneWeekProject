import numpy as np

class CTRV():
    def __init__(self, dt=0.1):

        """
        x : [x, y, v, theta, theta_rate]
        """


        self.dt = dt

    def step(self, x):

        if np.abs(x[4])>0.1:
            self.x = [x[0]+x[2]/x[4]*(np.sin(x[3]+x[4]*self.dt)-
                                                     np.sin(x[3])),
                      x[1]+x[2]/x[4]*(-np.cos(x[3]+x[4]*self.dt)+
                                                     np.cos(x[3])),
                      x[2],
                      x[3]+x[4]*self.dt,
                      x[4]]

        else:
            self.x = [x[0]+x[2]*np.cos(x[3])*self.dt,
                      x[1]+x[2]*np.sin(x[3])*self.dt,
                      x[2],
                      x[3],
                      x[4]]

        return self.x

    def H(self,x):

        return np.array([x[0],x[1],x[2],x[3]])

    def JA(self,x,dt = 0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        yaw = x[3]
        r = x[4]

        if np.abs(r)>0.1:
            # upper
            JA_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, v/r*(np.cos(yaw+r*dt)-np.cos(yaw)),
                     -v/(r**2)*(np.sin(yaw+r*dt)-np.sin(yaw))+v/r*(dt*np.cos(yaw+r*dt))],
                    [0, 1, (np.sin(yaw+r*dt)-np.sin(yaw))/r, v/r*(np.cos(yaw+r*dt)-np.cos(yaw)),
                     -v/(r**2)*(np.sin(yaw+r*dt)-np.sin(yaw))+v/r*(dt*np.cos(yaw+r*dt))],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 1]]
        else:
            JA_ = [[1, 0 , np.cos(yaw)*dt, -v*np.sin(yaw)*dt ,0],
                   [0, 1 , np.sin(yaw)*dt, v*np.cos(yaw)*dt,0],
                   [0,0,1,0,0],
                   [0,0,0,1,dt],
                   [0,0,0,0,1]]

        return np.array(JA_)

    def JH(self,x,dt = 0.1):


        JH_ = [[1,0,0,0,0],
               [0,1,0,0,0],
               [0,0,1,0,0],
               [0,0,0,1,0]]

        return np.array(JH_)


    def pred(self, x,  t_pred):

        self.x = x

        x_list = [self.x]
        for t in range(int(t_pred/self.dt)):
            x_list.append(self.step(self.x))

        return np.array(x_list)