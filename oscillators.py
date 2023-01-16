"""This is just a big copy-paste from Alan's code:
https://github.com/18alantom/synth  and 
https://python.plainenglish.io/build-your-own-python-synthesizer-part-2-66396f6dad81 
A few docstrings, etc have been added for use with gemusic.ipynb.
I could eventually just move to using Alan's code as a git submodule in this folder.
CW
"""
from abc import ABC, abstractmethod
import math
import itertools
import numpy as np
from scipy.io import wavfile

"""Oscillators"""
class Oscillator(ABC):
    """
    This is an iterator (__iter__ and __next__ are defined), which allows parameters such as freq, amp, phase, etc to be adjusted on-the-fly, without having to re-create a generator expression.
    The idea is that when a key is pressed __iter__ is called once, and the __next__ is called as long as the key is held.

    This abstract base class (ABC) can't be instantiated by itself unless its abstract methods are all defined.
    This is what the individual oscillators (below) will accomplish.
    Also, a static method squish_val has been added, this is to bring the oscillator output into a given range.

    Source:
    https://python.plainenglish.io/making-a-synth-with-python-oscillators-2cb8e68e9c3b
    """
    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1)):
        self._freq = freq # fundamental frequency
        self._amp = amp
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range
        
        # Properties that will be changed
        self._f = freq # alternate frequency
        self._a = amp
        self._p = phase
        self._i = 0
        
    @property
    def init_freq(self):
        return self._freq
    
    @property
    def init_amp(self):
        return self._amp
    
    @property
    def init_phase(self):
        return self._phase
    
    @property
    def freq(self):
        return self._f
    
    @freq.setter
    def freq(self, value):
        self._f = value
        self._post_freq_set()
        
    @property
    def amp(self):
        return self._a
    
    @amp.setter
    def amp(self, value):
        self._a = value
        self._post_amp_set()
        
    @property
    def phase(self):
        return self._p
    
    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()
    
    def _post_freq_set(self):
        pass
    
    def _post_amp_set(self):
        pass
    
    def _post_phase_set(self):
        pass
    
    @abstractmethod
    def _initialize_osc(self):
        pass
    
    @staticmethod
    def squish_val(val, min_val=0, max_val=1):
        return (((val + 1) / 2 ) * (max_val - min_val)) + min_val

    @abstractmethod
    def __next__(self):
        return None
    
    def __iter__(self):
        self.freq = self._freq
        self.phase = self._phase
        self.amp = self._amp
        self._initialize_osc()
        return self


class SineOscillator(Oscillator):
    """
    Derived class, implementing a sine wave generator.
    """
    def _post_freq_set(self):
        self._step = (2 * math.pi * self._f) / self._sample_rate
        
    def _post_phase_set(self):
        self._p = (self._p / 360) * 2 * math.pi
        
    def _initialize_osc(self):
        self._i = 0
        
    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a


class SquareOscillator(SineOscillator):
    """
    Derived class, implementing a square wave generator.
    Idea: threshold the sinewave 
    at some level and then return a high or a low value depending on 
    which side of the threshold the sine value is.
    
    Pretty harsh-sounding
    """
    def __init__(self, freq=440, phase=0, amp=1, 
                 sample_rate=44_100, wave_range=(-1, 1), threshold=0):
        super().__init__(freq, phase, amp, sample_rate, wave_range)
        self.threshold = threshold
    
    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if val < self.threshold:
            val = self._wave_range[0]
        else:
            val = self._wave_range[1]
        return val * self._a


class SawtoothOscillator(Oscillator):
    """
    Derived class, implementing a sine wave generator.
    """
    def _post_freq_set(self):
        self._period = self._sample_rate / self._f
        self._post_phase_set
        
    def _post_phase_set(self):
        self._p = ((self._p + 90)/ 360) * self._period
    
    def _initialize_osc(self):
        self._i = 0
    
    def __next__(self):
        div = (self._i + self._p )/self._period
        val = 2 * (div - math.floor(0.5 + div))
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a


class TriangleOscillator(SawtoothOscillator):
    """
    Sounds like a slightly dirty sine wave.
    Basically the absolute value of the sawtooth wave.
    """
    def __next__(self):
        div = (self._i + self._p)/self._period
        val = 2 * (div - math.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

"""Composers"""
class WaveAdder:
    """
    Add multiple oscillators together.
    """
    def __init__(self, *oscillators):
        self.oscillators = oscillators
        self.n = len(oscillators)
    
    def __iter__(self):
        [iter(osc) for osc in self.oscillators]
        return self
    
    def __next__(self):
        return sum(next(osc) for osc in self.oscillators) / self.n


"""Oscillator convenience functions"""
def get_val(oscillator, count=44100, it=False):
    """Returns 1 sec of samples of given osc at a given sample rate.
    Currently returns a python list, it could make sense to return a numpy array or other.
    """
    if it: oscillator = iter(oscillator)
    return [next(oscillator) for i in range(count)]


def fplot_xy(wave, fslice=slice(0,100), sample_rate=44100):
    """Take the FFT of a given waveform and retun xy values for a pyplot."""
    fd = np.fft.fft(wave)
    fd_mag = np.abs(fd)
    x = np.linspace(0, sample_rate, len(wave))
    y = fd_mag * 2 / sample_rate
    return x[fslice], y[fslice]


"""Modulators"""
class ADSREnvelope:
    """
    The simplest envelope.  It has 4 stages, explained below in terms of volume (amplitude),
    but note, the envelope could be used to modulate ANY of the oscillator parameters, 
    such as frequency, or phase ...

    - Attack : the time taken for a note to go from 0 to full volume. 
      Example : for plucked and percussive instruments this time taken is instant, 
      but say for something like a theremin, the rise to full volume can be much slower.

    - Decay : the time taken to reach the sustain level. 
      Example: for percussive sounds the decay is instant, i.e. these are transient 
      sounds, instant high amplitude for a very short amount of time.

    - Sustain : the level at which a note is held. 
      Example: for acoustic instruments the sustain will have decreasing amplitude which is why, 
      on the piano, the note will eventually die out. On electric guitars we can have infinite 
      sustain by using cool contraptions such as ebows or sustainer pickups. Digital instruments, 
      unconstrained by physics, can have infinite sustain.
    
    - Release : the time taken for the note to die out after it's released. 
      Example: when the finger is raised off of a piano key the volume doesn't instantly 
      drop to zero.

    A few complications:
    1. We don't know the length of the sustain stage, a note can be held for any amount of time.
    2. The release stage is triggered only when the note is released, so we'll need a way to 
       indicate that.
    3. The note can be released at any point, i.e. the envelope can be in the middle of the 
       attack or decay stages when the note is released, so we need to keep track of the current 
       value of the envelope to calculate the release values.

    The idea behind the ADSREnvelope class is that, when a note is pressed/played/activated, 
    __iter__ is called on it, and when it's released trigger_release is called. The envelope 
    steps through all of the values by calling __next__ on it until ended is set to True, i.e. 
    it is an iterator similar to the Oscillator class.
    """
    def __init__(self, attack_duration=0.05, decay_duration=0.2, sustain_level=0.7, \
                 release_duration=0.3, sample_rate=44100):
        self.attack_duration = attack_duration
        self.decay_duration = decay_duration
        self.sustain_level = sustain_level
        self.release_duration = release_duration
        self._sample_rate = sample_rate
        
    def get_ads_stepper(self):
        """
        Note : I'm calling them steppers cause they are stepping through the values of the envelope.
        
        Since we will be using itertools.count we need a step size, which here would be 
        the reciprocal of the number of samples in a stage.
        And depending on whether it is in the attack stage, or in either of the release 
        or decay stages, the step will be positive or negative respectively. 
        Sustain is constant so we can use itertools.cycle for this.

        Returns: A generator function that generates values for the attack, decay, and sustain stages.
        """
        steppers = []
        if self.attack_duration > 0:
            steppers.append(itertools.count(start=0, \
                step= 1 / (self.attack_duration * self._sample_rate)))
        if self.decay_duration > 0:
            steppers.append(itertools.count(start=1, \
            step=-(1 - self.sustain_level) / (self.decay_duration  * self._sample_rate)))
        while True:
            l = len(steppers)
            if l > 0:
                val = next(steppers[0])
                if l == 2 and val > 1:
                    steppers.pop(0)
                    val = next(steppers[0])
                elif l == 1 and val < self.sustain_level:
                    steppers.pop(0)
                    val = self.sustain_level
            else:
                val = self.sustain_level
            yield val
    
    def get_r_stepper(self):
        """
        Return: a generator function for the release stage.
        """
        val = 1
        if self.release_duration > 0:
            release_step = - self.val / (self.release_duration * self._sample_rate)
            stepper = itertools.count(self.val, step=release_step)
        else:
            val = -1
        while True:
            if val <= 0:
                self.ended = True
                val = 0
            else:
                val = next(stepper)
            yield val
    
    def __iter__(self):
        """ 
        We have to switch between the steppers at specific points:
          - Attack stepper should stop when amplitude reaches 1.
          - Decay stepper should stops when amplitude reaches the sustain level.
          - Sustain stepper should stop when the note is released.
          - Release stepper should stop when amplitude reaches 0.

        For the first 3 steppers we can create a generator function. But it wouldn't suffice 
        to take into account the third point and switch to the release_stepper, and also 
        there needs to be some indication of when release stage has ended.
        """
        self.val = 0
        self.ended = False
        self.stepper = self.get_ads_stepper()
        return self
    
    def __next__(self):
        self.val = next(self.stepper)
        return self.val
        
    def trigger_release(self):
        self.stepper = self.get_r_stepper()


"""Modulator convenience functions"""
def get_adsr(a=0.05, d=0.3, sl=0.7, r=0.2, sd=0.4, sample_rate=44_100):
    adsr = ADSREnvelope(a, d, sl, r)
    down_len = int(sum([a, d, sd]) * sample_rate)
    up_len = int(r * sample_rate)
    adsr = iter(adsr)
    adsr_vals = get_val(adsr, down_len)
    adsr.trigger_release()
    adsr_vals.extend(get_val(adsr, up_len))
    return adsr_vals, down_len, up_len


to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1, sample_rate=44_100):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    wavfile.write(fname, sample_rate, wav)


"""Modulated Oscillator"""
class ModulatedOscillator:
    """
    Creates a modulated oscillator by using a plain oscillator along with modulators,
    the `[parameter]_mod` functions of the signature (float, float) -> float are used
    to decide the method of modulation.
    Has `.trigger_release()` implemented to trigger the release stage of any of the modulators.
    similarly has `.ended` to indicate the end of signal generator of the modulators if the
    generation is meant to be finite.
    The ModulatedOscillator internal values are set by calling __init__ and then __next__
    to generate the sequence of values.
    """

    def __init__(
        self, oscillator, *modulators, amp_mod=None, freq_mod=None, phase_mod=None
    ):
        """
        oscillator : Instance of `Oscillator`, a component that generates a
            periodic signal of a given frequency.
        modulators : Components that generate a signal that can be used to
            modify the internal parameters of the oscillator.
            The number of modulators should be between 1 and 3.
            If only 1 is passed then then the same modulator is used for
            all the parameters.
        amp_mod : Any function that takes in the initial oscillator amplitude
            value and the modulator value and returns the modified value.
            If set the first modualtor is used for the values.
        freq_mod : Any function that takes in the initial oscillator frequency
            value and the modulator value and returns the modified value.
            If set the second modualtor of the last modulator is used for the values.
        phase_mod : Any function that takes in the initial oscillator phase
            value and the modulator value and returns the modified value.
            If set the third modualtor of the last modulator is used for the values.
        """
        self.oscillator = oscillator
        self.modulators = modulators
        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.phase_mod = phase_mod
        self._modulators_count = len(modulators)

    def __iter__(self):
        iter(self.oscillator)
        [iter(modulator) for modulator in self.modulators]
        return self

    def _modulate(self, mod_vals):
        if self.amp_mod is not None:
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_vals[0])
            self.oscillator.amp = new_amp

        if self.freq_mod is not None:
            if self._modulators_count == 2:
                mod_val = mod_vals[1]
            else:
                mod_val = mod_vals[0]
            new_freq = self.freq_mod(self.oscillator.init_freq, mod_val)
            self.oscillator.freq = new_freq

        if self.phase_mod is not None:
            if self._modulators_count == 3:
                mod_val = mod_vals[2]
            else:
                mod_val = mod_vals[-1]
            new_phase = self.phase_mod(self.oscillator.init_phase, mod_val)
            self.oscillator.phase = new_phase

    def trigger_release(self):
        tr = "trigger_release"
        for modulator in self.modulators:
            if hasattr(modulator, tr):
                modulator.trigger_release()
        if hasattr(self.oscillator, tr):
            self.oscillator.trigger_release()

    @property
    def ended(self):
        e = "ended"
        ended = []
        for modulator in self.modulators:
            if hasattr(modulator, e):
                ended.append(modulator.ended)
        if hasattr(self.oscillator, e):
            ended.append(self.oscillator.ended)
        return all(ended)

    def __next__(self):
        mod_vals = [next(modulator) for modulator in self.modulators]
        self._modulate(mod_vals)
        return next(self.oscillator)


"""Modulated Oscillator convenience functions"""
def amp_mod(init_amp, env):
    """
    For the oscillator amplitude it’s pretty simple, we just multiply the values
    """
    return env * init_amp
    

def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    """
    For frequency, we need to apply the envelope by only a small percent, 
    this will be set by the mod_amt parameter, and the sustain_level parameter 
    is so that when the note is in the sustain stage, it plays its actual frequency.

    phase_mod : For phase we can use the same function as freq_mod.
    """
    return init_freq + ((env - sustain_level) * init_freq * mod_amt)


def getdownlen(env, suslen, sample_rate=44_100):
    n = sum(env.attack_duration, env.release_duration, suslen)
    return int(n * sample_rate)


def gettrig(gen, downtime, sample_rate=44_100):
    gen = iter(gen)
    down = int(downtime * sample_rate)
    vals = get_val(gen, down)
    gen.trigger_release()
    while not gen.ended:
        vals.append(next(gen))
    return vals


def get_adsr_mod(a, d, sl, sd, r, Osc=SquareOscillator(55), mod = None):
    if mod is None:
        mod = ModulatedOscillator(
            Osc,
            ADSREnvelope(a,d,sl,r),
            amp_mod=amp_mod
        )
    downtime = a + d + sd
    return gettrig(mod, downtime)