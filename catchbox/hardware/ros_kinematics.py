"""KDL forward kinematics, built from the robot's published URDF."""
import numpy as np

class KDLKinematics:
    def __init__(self, base_link, tip_link, joint_names, opening_offset):
        import rospy, PyKDL
        from urdf_parser_py.urdf import URDF
        from kdl_parser_py.urdf import treeFromUrdfModel
        self.kdl=PyKDL;self.names=list(joint_names);self.offset=np.asarray(opening_offset,float)
        if self.offset.shape!=(3,) or not np.isfinite(self.offset).all():raise ValueError('opening_offset must be xyz in the tip frame')
        robot=URDF.from_parameter_server();ok,tree=treeFromUrdfModel(robot)
        if not ok:raise RuntimeError('Could not parse URDF into KDL')
        chain=tree.getChain(base_link,tip_link);names=[]
        for i in range(chain.getNrOfSegments()):
            j=chain.getSegment(i).getJoint()
            if j.getType()!=getattr(PyKDL.Joint, 'None'): names.append(j.getName())
        if names!=self.names:raise RuntimeError('KDL chain joints do not match configured joint order: '+str(names))
        self.solver=PyKDL.ChainFkSolverPos_recursive(chain)
        by_name={j.name:j for j in robot.joints}
        self.lower=np.array([by_name[n].limit.lower for n in self.names],float)
        self.upper=np.array([by_name[n].limit.upper for n in self.names],float)
    def position(self,q):
        a=self.kdl.JntArray(7)
        for i,x in enumerate(q):a[i]=float(x)
        frame=self.kdl.Frame()
        if self.solver.JntToCart(a,frame)<0:raise RuntimeError('KDL FK failed')
        p=frame*self.kdl.Vector(*self.offset)
        return np.array([p[0],p[1],p[2]])
