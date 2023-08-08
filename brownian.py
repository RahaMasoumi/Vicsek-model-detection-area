import numpy as np
from numpy import random
import basemodel as bm


class BrownianMotion(bm.AbstractTotalModel):
    """
    This class represents the Brownian motion of particles in 2 dimensions. The motion of the particles is restricted in
    a box. We consider the particles as ghost particles: they can intermingle.

    :param dt: Increment of time for each step. Constant * dt is the variance of the normal distribution that we use to calculate the increment of all the positions at each step.
    :type dt: float or int
    :param radius: radius of the particles. It as constant for all the particles
    :type radius: float or int
    :param contact_radius: distance from which we consider a contact between two particles
    :type contact_radius: float
    :param n_particles: Number of particles in the box
    :type n_particles: int
    :param surface: Surface of the box. We consider the box as a square, hence the length of the side is equal to the square root of the surface.
    :type surface: float or int
    :param n_steps: Number of steps that we consider for the total movement of the particles.
    :type n_steps: int
    """

    def __init__(self, diff, n_particles, dt=20, radius=.5, detection_radius=1, surface=10000, n_steps=20000, janus=False): 
        self.diff = diff
        super().__init__(n_particles, dt, radius, detection_radius, surface, n_steps, janus)
        self.position_array = np.random.rand(self.n_particles, 2) * self.side #create a (n_particles ,2) random array with elements between 0 and side (x, y) for each
        self.velocities_array = self.position_array
        self.detection_vector_array=self.position_array   #detection_area_model

    def brown_iter_2d(self):  #array of the increment of x and y. (dx, dy) array
        """Returns an array of the increment of the next position of the particles (dx, dy). As we consider a Brownian
        motion, the increment follows a 2D gaussian law of mean 0 and of variance dt.

        :return: Returns an array of the increment of the next position of the particles (dx, dy). It is of shape (n_particles, 2)
        :rtype: np.array
        """
        #return np.sqrt(self.diff * self.dt) * np.random.randn(self.n_particles, 2)
        r_max=np.sqrt(self.diff*self.dt)
        r=r_max*np.random.random(self.n_particles)
        theta=2*np.pi*random.random(self.n_particles)

        f_r=lambda r:r
        f_teta_cos=lambda theta: np.cos(theta)
        f_teta_sin=lambda theta: np.sin(theta)

        dx=f_r(r)*f_teta_cos(theta)
        dy=f_r(r)*f_teta_sin(theta)


        combined = np.vstack((dx, dy)).T
        return combined




    def brown_detect_vec_iter_2d(self):
        #In our detection area model we assign a detection area vector to each particles so that each particles can detect an area of 180 degree in front of itself 
        
        detect_vector_array=np.zeros((self.n_particles, 2))
        index=np.arange(0, self.n_particles)
        detect_vector_array[index, 0]=(np.random.rand(self.n_particles, 1).flatten() - 0.5) * 2 * 1

        alg = np.random.choice([-1, 1], self.n_particles)
        detect_vector_array[index, 1] = alg * np.sqrt(1 ** 2 - detect_vector_array[index, 0] ** 2)  #vy
        return detect_vector_array
    
################################################################################################################################


    def iter_movement(self, step, animation=False): # the array position at time step*dt
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a Brownian infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered directly in the function. If after iteration, x+dx or y+dy is inferior to 0 then the
        position is updated to x+dx=0 or y+dy=0 respectively. If after iteration, x+dx or y+dy is superior to self.side
        then the position is updated to x+dx=self.side or y+dy=self.side respectively.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        new_position = self.position_array + self.brown_iter_2d()
        new_position = np.where(new_position - self.radius <= 0, self.radius, new_position)
        new_position = np.where(new_position + self.radius >= self.side, self.side - self.radius, new_position)
        self.velocities_array = (new_position - self.position_array) / self.dt
        self.detection_vector_array=self.velocities_array
        self.position_array = new_position
        #self.detection_vector_array=self.brown_detect_vec_iter_2d()



        if not animation:
            contact_pairs, contact_index = self.contact()
            self.creation_tij(step, contact_pairs)  #what does this do? I should check what creation_tij will return (probably tij array)
