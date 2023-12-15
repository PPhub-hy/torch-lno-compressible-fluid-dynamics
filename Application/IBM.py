import torch
import numpy as np
import scipy.sparse as sp

class ClassicIBM:
    def __init__(self,h,p_e,p_l,p_ori,Length_x,Length_y,delta_s=0,NewBoundaryTreatment=True,If_Com=True):
        self.h = h
        self.delta_s = delta_s
        self.p_e = p_e
        self.p_l = p_l
        self.p_ori = p_ori
        self.Length_x = Length_x
        self.Length_y = Length_y
        self.xMax = np.max(p_e[:, 0])
        self.xMin = np.min(p_e[:, 0])
        self.yMax = np.max(p_e[:, 1])
        self.yMin = np.min(p_e[:, 1])
        self.NewBoundaryTreatment = NewBoundaryTreatment
        self.If_Com = If_Com

        xMax_l = np.zeros(len(self.p_ori))
        xMin_l = np.zeros(len(self.p_ori))
        yMax_l = np.zeros(len(self.p_ori))
        yMin_l = np.zeros(len(self.p_ori))
        self.Num_l = 0
        for ShapeOrder in range(len(self.p_ori)):
            xMax_l[ShapeOrder] = np.max(p_ori[ShapeOrder][:,0])
            xMin_l[ShapeOrder] = np.min(p_ori[ShapeOrder][:,0])
            yMax_l[ShapeOrder] = np.max(p_ori[ShapeOrder][:, 1])
            yMin_l[ShapeOrder] = np.min(p_ori[ShapeOrder][:, 1])
            self.Num_l += self.p_l[ShapeOrder].shape[0]-1
        self.IsInPoly = np.ones(self.p_e.shape[0], dtype='float32')
        self.InPolyOrder = np.ones(self.p_e.shape[0], dtype='int32') * (-1)
        self.IsAroundPoly = np.zeros(self.p_e.shape[0], dtype='float32')
        self.ClosestDist = np.ones(self.p_e.shape[0], dtype='int32') *100
        for i in range(self.p_e.shape[0]):
            for ShapeOrder in range(len(self.p_ori)):
                if self.p_e[i,0]>=xMin_l[ShapeOrder] - 2*self.h and self.p_e[i,0]<=xMax_l[ShapeOrder]+2*self.h and self.p_e[i,1]>=yMin_l[ShapeOrder]- 2*self.h and self.p_e[i,1]<=yMax_l[ShapeOrder]+ 2*self.h:
                    self.IsAroundPoly[i] = 1
                    if self.isPoiWithinPoly(self.p_e[i,:], ShapeOrder):
                        self.IsInPoly[i]=0
                        self.InPolyOrder[i] = ShapeOrder
        print('IsInPoly calculated!')

        for ShapeOrder in range(len(self.p_l)):
            self.p_l[ShapeOrder] = self.p_l[ShapeOrder][:-1, :]
        if self.NewBoundaryTreatment:
            self.GetNewMatrix()
            self.GetMatrix()
            print('IBM initialized!')
        else:
            self.GetMatrix()
            print('Classic IBM initialized!')


    def isRayIntersectsSegment(self, poi, s_poi, e_poi):
        if s_poi[1] == e_poi[1]:
            return False
        if s_poi[1] > poi[1] and e_poi[1] > poi[1]:
            return False
        if s_poi[1] < poi[1] and e_poi[1] < poi[1]:
            return False
        if s_poi[0] < poi[0] and e_poi[0] < poi[0]:
            return False
        if s_poi[1] == poi[1] and e_poi[1] > poi[1]:
            return False
        if e_poi[1] == poi[1] and s_poi[1] > poi[1]:
            return False

        xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])
        if xseg < poi[0]:
            return False
        else:
            return True

    def isPoiWithinPoly(self, poi, ShapeOrder):
        # 输入：点，多边形三维数组
        # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组
        sinsc = 0  # 交点个数
        for i in range(self.p_ori[ShapeOrder].shape[0]-1):  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
            if self.isRayIntersectsSegment(poi, self.p_ori[ShapeOrder][i,:], self.p_ori[ShapeOrder][i+1,:]):
                sinsc += 1  # 有交点就加1
        return True if sinsc % 2 == 1 else False

    def GetMatrix(self):
        # 按classic IBM的矩阵构建
        E2L = sp.coo_matrix((self.Num_l, self.p_e.shape[0]), ['float32']).tolil()
        L2E = sp.coo_matrix((self.p_e.shape[0], self.Num_l), ['float32']).tolil()
        for j in range(self.p_e.shape[0]):
            if self.IsAroundPoly[j] == 0:
                continue
            tempNum = -1
            tempnum = 0
            for ShapeOrder in range(len(self.p_l)):
                for i in range(self.p_l[ShapeOrder].shape[0]):
                    tempNum += 1
                    rx = abs(self.p_l[ShapeOrder][i, 0] - self.p_e[j, 0]) / self.h
                    if rx >= 2:
                        continue
                    ry = abs(self.p_l[ShapeOrder][i, 1] - self.p_e[j, 1]) / self.h
                    if rx < 2 and ry < 2:
                        if rx < 1:
                            dx = (3 - 2 * rx + np.sqrt(1 + 4 * rx - 4 * rx * rx)) / 8
                        elif rx < 2:
                            dx = (5 - 2 * rx - np.sqrt(-7 + 12 * rx - 4 * rx * rx)) / 8
                        if ry < 1:
                            dy = (3 - 2 * ry + np.sqrt(1 + 4 * ry - 4 * ry * ry)) / 8
                        elif ry < 2:
                            dy = (5 - 2 * ry - np.sqrt(-7 + 12 * ry - 4 * ry * ry)) / 8
                        E2L[tempNum, j] = dx * dy
                        L2E[j, tempNum] = dx * dy / self.h * self.delta_s
                        tempnum += dx * dy / self.h * self.delta_s

            L2E[j, :] = L2E[j, :] / tempnum
        self.E2L = E2L.tocsc().astype('float32')
        self.L2E = L2E.tocsc().astype('float32')

    def step(self,output):
        if self.If_Com:
            gamma = 1.4
            Ma = 0.2
            C_v=1/gamma/(gamma-1)/Ma/Ma
            u = output[0, 0, :, :].cpu().reshape(-1).numpy()
            v = output[0, 1, :, :].cpu().reshape(-1).numpy()

            rho = np.exp(output[0, 2, :, :].cpu().reshape(-1).numpy())
            #T = np.exp(output[0, 3, :, :].cpu().reshape(-1).numpy())
            #e = (u*u+v*v)/2+C_v*T

            RHOU = self.E2L.dot(rho*u)
            RHOV = self.E2L.dot(rho*v)
            RHO = self.E2L.dot(rho)

            fx = self.L2E.dot(-RHOU)
            fy = self.L2E.dot(-RHOV)
            delta_u = fx/rho
            delta_v = fy/rho
            u = (u + delta_u) * self.IsInPoly
            v = (v + delta_v) * self.IsInPoly

            return torch.from_numpy(u.reshape([self.Length_y + 1, self.Length_x + 1])).cuda(), \
                   torch.from_numpy(v.reshape([self.Length_y + 1, self.Length_x + 1])).cuda(), \
                   output[0, 2, :, :], \
                   output[0, 3, :, :]
        else:
            u = output[0, 0, :, :].cpu().reshape(-1).numpy()
            v = output[0, 1, :, :].cpu().reshape(-1).numpy()

            U = self.E2L.dot(u)
            V = self.E2L.dot(v)
            fx = self.L2E.dot(-U)
            fy = self.L2E.dot(-V)
            delta_u = fx
            delta_v = fy
            u = ((u + delta_u) * self.IsInPoly).reshape([self.Length_y + 1, self.Length_x + 1])
            v = ((v + delta_v) * self.IsInPoly).reshape([self.Length_y + 1, self.Length_x + 1])
            return torch.from_numpy(u).cuda(), torch.from_numpy(v).cuda()

    def GetLagrangeValue(self,u):
        return self.E2L.dot(u.cpu().reshape([-1,1]).numpy())

    def GetVerticalIntersec(self, poi, s_poi, e_poi):
        # 输入： 点，线段起点与终点
        # 输出： 垂线交点坐标，是否落在线段内，垂直距离（若在线段内）
        poi_intersec = np.zeros(2, dtype='float32')
        dist = -1
        InLine = False
        if abs(s_poi[0] - e_poi[0]) < 1e-6:  # x=const 垂直线段
            poi_intersec[0] = s_poi[0]
            poi_intersec[1] = poi[1]
            if poi[1] >= min(s_poi[1], e_poi[1]) and poi[1] <= max(s_poi[1], e_poi[1]):
                InLine = True
                dist = abs(poi[0] - s_poi[0])
        elif abs(s_poi[1] - e_poi[1]) < 1e-6:  # y=const 水平线段
            poi_intersec[1] = s_poi[1]
            poi_intersec[0] = poi[0]
            if poi[0] >= min(s_poi[0], e_poi[0]) and poi[1] <= max(s_poi[0], e_poi[0]):
                InLine = True
                dist = abs(poi[1] - s_poi[1])
        else:
            k1 = (s_poi[1] - e_poi[1]) / (s_poi[0] - e_poi[0])
            b1 = e_poi[1] - k1 * e_poi[0]
            k2 = -1 / k1
            b2 = poi[1] - k2 * poi[0]
            poi_intersec[0] = -(b1 - b2) / (k1 - k2)
            poi_intersec[1] = k1 * poi_intersec[0] + b1
            if poi_intersec[0] >= min(s_poi[0], e_poi[0]) and poi_intersec[0] <= max(s_poi[0], e_poi[0]):
                InLine = True
                dist = np.sqrt(np.sum((poi_intersec - poi) ** 2))
        return poi_intersec, InLine, dist

    def GetClosestLagrangePoint(self, poi, ShapeOrder):
        # 输入：某点坐标，所在的固壁编号
        # 输出： 距该点最近的Lagrange点编号
        dist = np.sqrt((poi[0] - self.p_l[ShapeOrder][:, 0]) ** 2 + (poi[1] - self.p_l[ShapeOrder][:, 1]) ** 2)

        return np.argmin(dist)

    def GetBoundaryIntercept(self, poi, ShapeOrder, Order_closest):
        # 输入：固壁内点坐标，所在的固壁编号，最近的Lagrange点编号
        # 输出：Boundary intercept点的坐标， image point的坐标
        if Order_closest == 0:
            poi_intersec1, InLine1, dist1 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][-1, :],
                                                                     self.p_l[ShapeOrder][Order_closest, :])
            poi_intersec2, InLine2, dist2 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][Order_closest, :],
                                                                     self.p_l[ShapeOrder][Order_closest + 1, :])
        elif Order_closest == self.p_l[ShapeOrder].shape[0] - 1:
            poi_intersec1, InLine1, dist1 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][Order_closest - 1, :],
                                                                     self.p_l[ShapeOrder][Order_closest, :])
            poi_intersec2, InLine2, dist2 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][Order_closest, :],
                                                                     self.p_l[ShapeOrder][0, :])
        else:
            poi_intersec1, InLine1, dist1 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][Order_closest - 1, :],
                                                                     self.p_l[ShapeOrder][Order_closest, :])
            poi_intersec2, InLine2, dist2 = self.GetVerticalIntersec(poi, self.p_l[ShapeOrder][Order_closest, :],
                                                                     self.p_l[ShapeOrder][Order_closest + 1, :])
        if InLine1 and not InLine2:
            return poi_intersec1, 2 * poi_intersec1 - poi, dist1
        elif InLine2 and not InLine1:
            return poi_intersec2, 2 * poi_intersec2 - poi, dist2
        elif InLine1 and InLine2:
            if dist1 > dist2:
                return poi_intersec2, 2 * poi_intersec2 - poi, dist2
            else:
                return poi_intersec1, 2 * poi_intersec1 - poi, dist1
        else:
            return self.p_l[ShapeOrder][Order_closest, :], 2 * self.p_l[ShapeOrder][Order_closest, :] - poi, np.sqrt((self.p_l[ShapeOrder][Order_closest, :]-poi)**2)

    def GetBilinearInterpolation(self, poi):
        I = int((poi[0] - self.xMin) // self.h)
        J = int((poi[1] - self.yMin) // self.h)

        if I < 0:
            I = 0
            poi[0] = self.p_e[I + (self.Length_x + 1) * J, 0]
        if J < 0:
            J = 0
            poi[1] = self.p_e[I + (self.Length_x + 1) * J, 1]
        if I>self.Length_x:
            I = self.Length_x
            poi[0] = self.p_e[I + (self.Length_x + 1) * J, 0]
        if J > self.Length_y:
            J = self.Length_y
            poi[1] = self.p_e[I + (self.Length_x + 1) * J, 1]
        assert I * self.h + self.xMin == self.p_e[I + (self.Length_x + 1) * J, 0]
        seta = ((poi[0]-self.xMin) / self.h - I) * 2 - 1
        ita = ((poi[1]-self.yMin) / self.h - J) * 2 - 1
        assert seta>=-1 and seta<=1
        assert ita>=-1 and ita<=1
        return np.array([(1 - seta) * (1 - ita) / 4, (1 + seta) * (1 - ita) / 4, (1 + seta) * (1 + ita) / 4,
                         (1 - seta) * (1 + ita) / 4]), np.array(
            [I + (self.Length_x + 1) * J, I + 1 + (self.Length_x + 1) * J, I + 1 + (self.Length_x + 1) * (J + 1),
             I + (self.Length_x + 1) * (J + 1)])

    def GetNewMatrix(self):
        # 按新边界处理方法的矩阵构建
        InterMatrix = sp.coo_matrix((self.p_e.shape[0], self.p_e.shape[0]), ['float32']).tolil()
        InterMatrix_u = sp.coo_matrix((self.p_e.shape[0], self.p_e.shape[0]), ['float32']).tolil()
        for i in range(self.p_e.shape[0]):
            if self.InPolyOrder[i] >= 0:
                Order_closest = self.GetClosestLagrangePoint(self.p_e[i, :], self.InPolyOrder[i])
                BoundaryIntercept, ImagePoint, dist = self.GetBoundaryIntercept(self.p_e[i, :], self.InPolyOrder[i],
                                                                          Order_closest)
                coeff, order = self.GetBilinearInterpolation(ImagePoint)
                d = 1
                InterMatrix[i, order] = coeff
        self.InterMatrix = InterMatrix.tocsc().astype('float32')

    def Newstep(self, output):
        u = output[0, 0, :, :].cpu().reshape(-1).numpy()
        v = output[0, 1, :, :].cpu().reshape(-1).numpy()
        rho = np.exp(output[0, 2, :, :].cpu().reshape(-1).numpy())
        T = np.exp(output[0, 3, :, :].cpu().reshape(-1).numpy())

        RHOU = self.E2L.dot(rho * u)
        RHOV = self.E2L.dot(rho * v)

        fx = self.L2E.dot(-RHOU)
        fy = self.L2E.dot(-RHOV)
        delta_u = fx / rho
        delta_v = fy / rho
        u = (u + delta_u) * self.IsInPoly
        v = (v + delta_v) * self.IsInPoly

        tol = 1
        times = 0
        while tol > 1e-5 and times < 1:
            times += 1
            u1 = u * self.IsInPoly
            v1 = v * self.IsInPoly
            rho1 = rho * self.IsInPoly + self.InterMatrix * (rho * T)
            T1 = T * self.IsInPoly + (1 - self.IsInPoly)
            tol = max(np.max(u1-u),np.max(v1-v))
            #print(times,tol)
            u = u1
            v = v1
            rho = rho1
            T = T1

        return torch.from_numpy(u.reshape([self.Length_y + 1, self.Length_x + 1])).cuda(),\
               torch.from_numpy(v.reshape([self.Length_y + 1, self.Length_x + 1])).cuda(),\
               torch.from_numpy(np.log(rho).reshape([self.Length_y + 1, self.Length_x + 1])).cuda(),\
               torch.from_numpy(np.log(T).reshape([self.Length_y + 1, self.Length_x + 1])).cuda()



