from matplotlib import pyplot as plt
from scipy.io import wavfile
import numpy as np
import IPython
from scipy.fftpack import fft,ifft,fft2,ifft2
from scipy.signal import fftconvolve
import time

class SoundWave(object):
    """A class for working with digital audio signals."""

    def __init__(self, rate, samples):
        """Set the SoundWave class attributes.

        Parameters:
            rate (int): The sample rate of the sound.
            samples ((n,) ndarray): NumPy array of samples.
        """
        #save attributes
        self.rate = rate
        self.samples = samples
        self.n = len(self.samples)

    def plot(self, xmin=None, xmax=None, frequency=False):
        """Plot the graph of the sound wave (time versus amplitude)."""
        #length is number of samples / sample rate
        length = len(self.samples)/self.rate
        
        #plot
        if frequency: 
            plt.subplot(121)
            plt.title('Time Domain')
#         plt.ylim(-32768,32767)
        plt.ylabel('Amplitude')
        plt.xlim(0,length)
        plt.xlabel('Time (seconds)')
        time = np.linspace(0,length,len(self.samples))
        plt.plot(time,self.samples)
        
        #plot frequency domain
        if frequency:
            plt.subplot(122)
            plt.title('Frequency Domain')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            DFT = fft(self.samples)
            n = len(self.samples)
            frequencies = [k*self.rate/n for k in range(n)]
            if xmin is not None:
                plt.xlim(xmin,xmax)
            plt.plot(frequencies[:n//2],abs(DFT[:n//2]))
            plt.tight_layout()

    def export(self, filename, force=False):
        """Generate a wav file from the sample rate and samples. 
        If the array of samples is not of type np.int16, scale it before exporting.

        Parameters:
            filename (str): The name of the wav file to export the sound to.
        """
        #just the real part
        samples = self.samples.real
        
        #rescale
        if type(samples[0]) is not np.int16 or force:
            samples = np.int16((samples/max(abs(samples)))*32767)
            
        #write
        wavfile.write(filename,self.rate,samples)
    
    def __add__(self, other):
        """Combine the samples from two SoundWave objects.

        Parameters:
            other (SoundWave): An object containing the samples to add
                to the samples contained in this object.
        
        Returns:
            (SoundWave): A new SoundWave instance with the combined samples.

        Raises:
            ValueError: if the two sample arrays are not the same length.
        """
        #value error
        if len(self.samples) != len(other.samples):
            raise ValueError('SoundWave objects must have the same length')
            
        return SoundWave(self.rate, self.samples+other.samples)

    def __rshift__(self, other):
        """Concatentate the samples from two SoundWave objects.

        Parameters:
            other (SoundWave): An object containing the samples to concatenate
                to the samples contained in this object.

        Raises:
            ValueError: if the two sample rates are not equal.
        """
        #value error
        if self.rate != other.rate:
            raise ValueError('SoundWave objects must have the same sampling rate')
            
        return SoundWave(self.rate, np.concatenate((self.samples,other.samples)))
    
    def __mul__(self, other):
        """Convolve the samples from two SoundWave objects using circular convolution.
        
        Parameters:
            other (SoundWave): An object containing the samples to convolve
                with the samples contained in this object.
        
        Returns:
            (SoundWave): A new SoundWave instance with the convolved samples.

        Raises:
            ValueError: if the two sample rates are not equal.
        """
        #value error
        if self.rate != other.rate:
            raise ValueError("SoundWave objects must have the same sampling rate")
            
        #pad shorter one
        Asamples = self.samples
        Bsamples = other.samples
        while len(Asamples) < len(Bsamples):
            Asamples = np.append(Asamples,[0])
        while len(Bsamples) < len(Asamples):
            Bsamples = np.append(Bsamples,[0])
            
        #return their convolution
        return SoundWave(self.rate,ifft(fft(Asamples)*fft(Bsamples)))
            

    def __pow__(self, other):
        """Convolve the samples from two SoundWave objects using linear convolution.
        
        Parameters:
            other (SoundWave): An object containing the samples to convolve
                with the samples contained in this object.
        
        Returns:
            (SoundWave): A new SoundWave instance with the convolved samples.

        Raises:
            ValueError: if the two sample rates are not equal.
        """
        #value error
        if self.rate != other.rate:
            raise ValueError("SoundWave objects must have the same sampling rate")
            
        #find a
        a = 0
        n = len(self.samples)
        m = len(other.samples)
        while 2**a < n + m - 1: a+=1
            
        #pad samples
        Asamples = np.append(self.samples,np.zeros(2**a-n))
        Bsamples = np.append(other.samples,np.zeros(2**a-m))
        
        convolution = ifft(fft(Asamples)*fft(Bsamples))
        return SoundWave(self.rate, convolution[:n+m-1])
        

    def clean(self, low_freq, high_freq):
        """Remove a range of frequencies from the samples using the DFT. 

        Parameters:
            low_freq (float): Lower bound of the frequency range to zero out.
            high_freq (float): Higher boound of the frequency range to zero out.
        """
        klow = int(low_freq*self.n/self.rate)
        khigh = int(high_freq*len(self.samples)/self.rate)
        
        DFT = fft(self.samples)
        for k in range(klow,khigh+1):
            DFT[k]=0
        for k in range(self.n-khigh,self.n-klow+1):
            DFT[k]=0
            
        self.samples = ifft(DFT).real

# a function to generate sine waves
def generate_note(frequency, duration):
    """Generate an instance of the SoundWave class corresponding to 
    the desired soundwave. Uses sample rate of 44100 Hz.
    
    Parameters:
        frequency (float): The frequency of the desired sound.
        duration (float): The length of the desired sound in seconds.
    
    Returns:
        sound (SoundWave): An instance of the SoundWave class.
    """
    rate = 44100
    f = lambda x: np.sin(2*np.pi*x*frequency)
    samples = np.arange(rate*duration)/rate
    return SoundWave(rate,f(samples))
