import numpy as np
from neuropy.utils import mathutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from neuropy.core.epoch import Epoch
from neuropy.core.datawriter import DataWriter

from neuropy.core import Position
from neuropy.core import Epoch
from neuropy.utils.mathutil import thresh_epochs

class CircularPosition(Position):
    """ Inherits from Position:
    x
    y
    z
    t_start
    n_frames
    duration
    t_stop
    time
    ndim
    sampling_rate
    speed (overwritten below to account for wrapping)
    get_smoothed (overwritten to produce CircularPosition obj)
    to_dataframe
    from_dataframe
    speed_in_epochs (I don't think this is functional? jan 2025)
    time_slice (overwritten to produce CircularPosition obj)
      """
    
    def __init__(
        self, *args,**kwargs):

        super().__init__(*args, **kwargs)

        # if isinstance(traces_rot,np.ndarray):
        #     if traces_rot.ndim == 1:
        #         traces_rot = traces_rot.reshape(1,-1)
        #     assert traces_rot.shape[0] <= 3, "Maximum possible dimension of rotation is 3"
        #     self.traces_rot =  traces_rot

        # self._speed = None #for manual overwrite

            ### overwriting here in order to have the setter.
    @property
    def speed(self):
        #if self._speed is not None:  # Return overridden speed if set
        #     return self._speed

        if self.ndim == 1: #get linear speed
            speed = self.get_linear_speed() #returns linear
            speed_to_return = np.hstack(speed)
        else:
            # Compute speed from traces if not manually set
            dt = 1 / self.sampling_rate
            speed = np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt
            speed_to_return = np.hstack(([0], speed))

        return speed_to_return

    # @speed.setter #to overwrite, if you want consistent speed between 2d and 1d for ex
    # def speed(self, speed_overwrite):
    #     if len(speed_overwrite) != self.traces.shape[1]:
    #         raise ValueError("speed_overwrite must match the number of time points in traces.")
        
    #     self._speed = speed_overwrite  # Store overridden speed

    #@property
    def get_linear_speed(self):
        #assumes 1D
        
        dt = 1 / self.sampling_rate

        d = np.diff(self.traces[0])
        d[d>6] -= 2*np.pi #manual unwrap cause np.unwrap wasn't looking right?
        d[d<-6] += 2*np.pi

        speed = np.sqrt(np.abs(d) ** 2) / dt
        # speed = np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt

        return np.hstack(([0], speed))
    
    import numpy as np

    @property
    def rotational_velocity(self):

        dt = 1/self.sampling_rate

        xr = self.x_rot #pitch
        yr = self.y_rot #yaw
        zr = self.z_rot #roll
        # Compute derivatives (finite difference)
        dxr_dt = np.gradient(xr, dt)
        dyr_dt = np.gradient(yr, dt)
        dzr_dt = np.gradient(zr, dt)

        # Convert angles to radians
        xr = np.radians(xr)
        yr = np.radians(yr)
        zr = np.radians(zr)

        # Compute angular velocity for each timestep
        omega_x = dxr_dt - np.sin(yr) * dzr_dt
        omega_y = np.cos(xr) * dyr_dt + np.cos(yr) * np.sin(xr) * dzr_dt
        omega_z = -np.sin(xr) * dyr_dt + np.cos(yr) * np.cos(xr) * dzr_dt

        rotational_velocity = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

        return np.hstack(rotational_velocity)
    
    def rescale(self,
                xbounds:list=None,
                zbounds:list=None,
                center=False,
                fullsize=182.88):
        # to reshape circular track into an exact circle, 6ft outer to outer rims

        # ex usage:
        # xbounds = [-86.2,87.2]
        # zbounds = [-87.8,85.7]
        # pos_track.rescale(xbounds=xbounds,zbounds=zbounds,center=True)

        x_curr = xbounds[1]-xbounds[0]
        z_curr = zbounds[1]-zbounds[0]

        x_factor = fullsize/x_curr
        z_factor = fullsize/z_curr
        y_factor = (x_factor + z_factor)/2 #estimate

        new_x = self.x*x_factor
        new_z = self.z*z_factor
        new_y = self.y*y_factor

        if center:
            xbounds_new = np.array(xbounds)*x_factor
            zbounds_new = np.array(zbounds)*z_factor
            xcenter = sum(xbounds_new)/2
            zcenter = sum(zbounds_new)/2

           # xcenter = np.nanmedian([np.nanmin(new_x),np.nanmax(new_x)])
           # zcenter = np.nanmedian([np.nanmin(new_z),np.nanmax(new_z)])
            new_x = new_x-xcenter
            new_z = new_z-zcenter
        
        self.x = new_x
        self.z = new_z
        self.y = new_y

        if self.traces_rot is not None:

            new_x_rot = self.x_rot * x_factor
            new_z_rot = self.z_rot * z_factor
            new_y_rot = self.y_rot * y_factor

            self.x_rot = new_x_rot
            self.z_rot = new_z_rot
            self.y_rot = new_y_rot

            return CircularPosition(
                traces=self.traces,
                traces_rot=self.traces_rot,
                sampling_rate=self.sampling_rate,
                t_start=self.t_start,
            )
        else:
            return CircularPosition(
            traces=self.traces,
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
        )

    def get_linearized(self, dimensions=["x", "z"],track_radius = 91.44,track_width = 5,width_buffer = 3):
        """
    Parameters
    ----------
    position: circularPosition (core.Position)
        Position object containing spatial information
    dimensions: list, optional
        List of spatial dimensions to use, by default ["x", "z"].
    track_radius: in cm. default is for 6 ft diameter track (outer edge)
    track_width: in cm.

    Assumptions
    ---------
    circular track
    track has been rescaled to 6 ft diameter with equal x/y scaling (see self.rescale)

    Returns
    -------
    self, as a new circularPosition object with linearized traces.
    """
        pos_components = []
        for dim in dimensions:
            if hasattr(self, dim):
                pos_components.append(getattr(self, dim))
            else:
                raise ValueError(f"Dimeinos '{dim}' not found in the position object.")

        # Combined dimensions
        pos_array = np.vstack(pos_components).T

        theta = None

        #need to get xlim,ylim, set 0 as the center of those
        x, y = pos_array[:, 0], pos_array[:, 1]
        xcenter = np.nanmean([np.nanmin(x),np.nanmax(x)])
        ycenter = np.nanmean([np.nanmin(y),np.nanmax(y)])
        x = x-xcenter
        y = y-ycenter

        theta = np.arctan2(y,x);

        #get distance from center for filtering out off-track position
        distance = np.sqrt(x**2 + y**2)

        # Apply mask to get theta where position is on track
        outer_edge = track_radius + width_buffer # +2 for leeway
        inner_edge = track_radius - track_width - width_buffer
        on_track = (distance > inner_edge) & (distance < outer_edge)
        theta = np.where(on_track, theta, np.nan)

        theta[theta < 0] += 2*np.pi; #have all between 0 and 2pi

        return CircularPosition(
            traces=theta, t_start=self.t_start, sampling_rate=self.sampling_rate
        )

    def get_smoothed(self, sigma):
        dt = 1 / self.sampling_rate
        smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)

        if self.traces_rot is not None:
            return CircularPosition(
                traces=smooth(self.traces),
                traces_rot=smooth(self.traces_rot),
                sampling_rate=self.sampling_rate,
                t_start=self.t_start,
            )
        else:
            return CircularPosition(
            traces=smooth(self.traces),
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
        )

    def time_slice(self, t_start, t_stop, zero_times=False):
        indices = super()._time_slice_params(t_start, t_stop)
        if zero_times:
            t_stop = t_stop - t_start
            t_start = 0

        if self.traces_rot is not None:
            return CircularPosition(
                traces=self.traces[:,indices],
                traces_rot=self.traces_rot[:,indices],
                sampling_rate=self.sampling_rate,
                t_start=t_start,
            )
        else:
            return CircularPosition(
                traces=self.traces[:, indices],
                t_start=t_start,
                sampling_rate=self.sampling_rate,
            )


    def get_run_direction_epochs(
        self,
        speed_thresh=(0.3, None),
        speed_2D:np.ndarray = None,
        boundary=np.pi/15, #8.0,
        duration=(0.5, None),
        sep=0.5,
        min_distance=np.pi/30, #10,
        sigma=0.05,
        unwrapped = None
    ):
        """Divide running epochs into up (increasing values) and down (decreasing values).
        Currently only works for one dimensional position data

        Parameters
        ----------
        speed_thresh : tuple, optional
            low and high speed threshold for speed, by default (10, 20) in cm/s
        boundary: float
            boundaries of epochs are extended to this value, in cm/s
        duration : int, optional
            min and max duration of epochs, in seconds
        sep: int, optional
            epochs separated by less than this many seconds will be merged
        min_distance : int, optional
            the animal should cover this much distance in one direction within the lap to be included, by default 50 cm
        sigma : int, optional
            speed is smoothed, increase if epochs are fragmented, by default 10
        plot : bool, optional
            plots the epochs with position and speed data, by default True
        """

        metadata = locals()
        #metadata.pop("position")
        assert self.ndim == 1, "Run direction only supports one dimensional position"

        sampling_rate = self.sampling_rate
        dt = 1 / sampling_rate

        if unwrapped is not None:
            x = unwrapped
        else:
            x = self.x

        if speed_2D is not None: #if you want to pass in 2D speed for consistency.
            speed = speed_2D
        else: #otherwise use linear speed. make sure speed thresh matches your input type!
            speed = gaussian_filter1d(self.speed, sigma=sigma / dt)

        #get all epochs, irregardless of direction
        starts, stops, peak_time, peak_speed = thresh_epochs(
            arr=speed,
            thresh=speed_thresh,
            length=duration,
            sep=sep,
            boundary=boundary,
            fs=sampling_rate,
        )

        high_speed = np.vstack((starts, stops)).T
        high_speed = high_speed * sampling_rate  # convert to index locations
        val = []
        for epoch in high_speed.astype("int"):
            displacement = x[epoch[1]] - x[epoch[0]]
            # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

            if np.abs(displacement) > min_distance:
                if displacement < 0:
                    val.append(-1)
                elif displacement > 0:
                    val.append(1)
            else:
                val.append(0)
        val = np.asarray(val)

        # ---- deleting epochs where animal ran a little distance------
        ind_keep = val != 0
        high_speed = high_speed[ind_keep, :]
        val = val[ind_keep]
        peak_time = peak_time[ind_keep]
        peak_speed = peak_speed[ind_keep]

        high_speed = np.around(high_speed / sampling_rate + self.t_start, 2)
        data = pd.DataFrame(high_speed, columns=["start", "stop"])
        data["label"] = np.where(val > 0, "up", "down")
        data["peak_time"] = peak_time + self.t_start
        data["peak_speed"] = peak_speed

        return Epoch(epochs=data, metadata=metadata)
    
    def get_unwrapped(self):
        
        assert self.ndim == 1, "Run direction only supports one dimensional position"

        diff = np.diff(self.x)
        jump_ind = np.where(abs(diff) > 6)[0]
        counter = diff[jump_ind] > 0 # in the counterclockwise dir - double check this in position! 

        unwrapped = self.x.copy()

        for i in range(len(counter)): #[0:len(counter)-1]:#range(0,len(counter)-1):
            if counter[i] == True:
                unwrapped[jump_ind[i]+1:] -= 2*np.pi
            else:
                unwrapped[jump_ind[i]+1:] += 2*np.pi

        return unwrapped #vector


    