import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib.adni.seqVisualizer import seqVisualizer


class Visualizer(seqVisualizer):
    def __init__(self, frame_size, slices, ncols=4, figsize=(12, 8)):

        self.slices = slices  # x,y,z slices - vector up to 3 elements
        self.nrows = len(self.slices)
        self.ncols = ncols

        self.frame_size = frame_size
        self.fig, self.axarr = plt.subplots(self.nrows,
                                            self.ncols,
                                            figsize=figsize)
        fig_arr = [[mpimg.AxesImage for j in range(self.ncols)]
                   for i in range(self.nrows)]
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                rand_mat = np.random.random(size=self.frame_size)
                self.axarr[ii, jj].axis('off')
                fig_arr[ii][jj] = self.axarr[ii, jj].imshow(rand_mat,
                                                            cmap='gray')

    def make_plot(self, data):
        for slice_iter, slice_ in enumerate(self.slices):
            for c in range(self.ncols):
                tmp = np.reshape(data[c, :].T, data.shape[-3:])
                tmp = np.flipud(np.squeeze(np.take(tmp, slice_, slice_iter)))

                self.axarr[slice_iter, c].clear()
                self.axarr[slice_iter, c].imshow(
                    tmp,
                    #vmin=np.min(tmp),  #0
                    #vmax=np.max(tmp),
                    clim=(0.0, 1.0))
                self.axarr[slice_iter, c].axis('off')
