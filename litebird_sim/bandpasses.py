import litebird_sim as lbs
import numpy as np
from scipy import signal


@dataclass
class  BandPassInfo(object) :

    """A class wrapping the basic information about a detector bandpass.
    This  class   encodes the basic properties of a frequency band.
    It can be initialized in three ways:
    - Through the default constructor, will assume top-hat bandpasses
    provided the band centroid and the width are issued.
    - Through the class method :meth:`.from_imo`, which reads the
      detector bandpass from the LiteBIRD Instrument Model (see
      the :class:`.Imo` class).

    Args:
        - nu_center (float):  center frequency in GHz
        - bandwidth (float):  width of the band (default=0.2)
        - range_offset (int): increase the freq. range by an offset (default +- 10 from the edges)
        - nsamp (int): number of elements to sample the band (default=128)
        - name (str) : ID of the band
        - normalize(bool) : If set to true bandpass weights will be normalized to 1
        - bandtype (str): choose between:
            - "top-hat" (default)
            - "top-hat-exp" : the edges of the band are apodized with an exponential profile
            - "top-hat-cosine" : the edges of the band are apodized with a cosine profile
            - "cheby" : the bandpass encodes a _Chebyshev_  profile
    """


    bandcenter_ghz : float = 0.0
    bandwidth_ghz: float = 0.0
    range_offset: int = 10
    nsamp : int =128
    name : str =""
    bandtype : str ="top-hat"
    normalize : bool = False


    def __post_init__(self ):
        """
        Constructor of the class

        """

        self.f0, self. f1  = self.get_edges ()

        # we extend the wings of the top-hat bandpass  with 10 samples
        #before  and after the edges
        bandrange =self. f0 -range_offset ,self. f1 +range_offset
        self.freqs  = np.linspace(bandrange[0], bandrange[1], nsamp )
        self.isnormalized = False
        if self.bandtype == "top-hat" :
            self.get_top_hat_bandpass(normalize=self.normalize )
        elif self.bandtype == "top-hat-cosine" :
            self.get_top_hat_bandpass(apodization='cosine',normalize=self.normalize)
        elif self.bandtype == "top-hat-exp" :
            self.get_top_hat_bandpass(apodization='exp',normalize=self.normalize)
        elif self.bandtype == "cheby" :
            self.get_chebyshev_bandpass(normalize=self.normalize)
        else:
            print( f"{self.bandtype} profile not implemented. Assume top-hat")
            self.get_top_hat_bandpass(normalize=self.normalize)


    def get_edges (self ) :
        """
        get the edges of the tophat band
        """
        return self.bandcenter_ghz*(1- self.bandwidth_ghz /2),self.bandcenter_ghz*(1+self.bandwidth_ghz/2)




    def get_top_hat_bandpass( self  ,
                             normalize= False  , apodization= None  ):
        """
        Sample  a top-hat bandpass, givn the centroid and the bandwidth
        normalize:
        normalize the transmission coefficients so that its integral = 1 over the freq.  band
        apodization:
        if None no apodization is applied to the edges, otherwise a string between `cosine` or `exp` will
        apodize the edges following the chosen  profile
        """


        self. weights = np.zeros_like(self.freqs)
        mask = np.ma.masked_inside( self.freqs,self.f0,self.f1 ).mask
        self. weights [mask] = 1.

        if apodization == 'cosine' :
            #print(f"Apodizing w/ {apodization} profile")
            self. cosine_apodize_bandpass()

        elif apodization == 'exp':
            #print(f"Apodizing w/ {apodization} profile")
            self. exp_apodize_bandpass()

        elif  apodization   is None:
            #print("/!\ Band is not apodized")
        if normalize :
            self.normalize_band()


    def normalize_band(self):
        """
        Normalize the band transmission coefficients

        """
        A = np.trapz(self. weights , self.freqs )
        self. weights  /=A
        self.isnormalized=True



    def exp_apodize_bandpass( self,  alpha = 1, beta = 1):
        """Define a bandpass with exponential tails and unit transmission in band
        freqs: frequency in GHz

        Args:
        - alpha (float): out-of-band exponential decay index for low freq edge
        - beta (float) : out-of-band exponential decay index for high freq edge

        If alpha and beta are not specified a value of 1 is used for both.

        """
        mask_beta=np.ma.masked_greater(self.freqs,self.f1 ).mask
        self.weights[mask_beta] = np.exp(-beta * (self.freqs[mask_beta] - self.f1))
        mask_alpha = np.ma.masked_less( self.freqs, self.f0 ).mask
        self. weights[mask_alpha] = np.exp(alpha * (self.freqs [mask_alpha] -self. f0))

    def cosine_apodize_bandpass( self, a = 5  ):
        """
        Define a bandpass with cosine tails and unit transmission in band

        Args:
        - a (int): the numerical factor related to the apodization length

        """

        apolength =   self.bandwidth_ghz/a
        apod = lambda x, a,b: (1 + np.cos((x-a)/(b-a)  *np.pi ))/2
        f_above= self.bandcenter_ghz * ( 1 + self.bandwidth_ghz/2 + apolength )
        f_below= self.bandcenter_ghz * ( 1 - self.bandwidth_ghz/2 - apolength )
        mask_above=np.ma.masked_inside(self.freqs,self.f1 , f_above  ).mask

        x_ab = np.linspace(self.f1 , f_above,self.freqs[mask_above].size )

        self.weights[mask_above] = apod(x_ab,self.f1 , f_above)

        mask_below=np.ma.masked_inside(self.freqs, f_below, self.f0 ).mask
        x_bel  = np.linspace(f_below, self.f0,self.freqs[mask_below].size )
        self.weights[mask_below] = apod( x_bel, self.f0 , f_below)

    # Chebyshev profile bandpass
    def get_chebyshev_bandpass(self,  order = 3, ripple_dB = 3, normalize=False ):
        """
        Define a bandpass with chebyshev prototype.

        Args:
        - order (int): chebyshev filter order
        - ripple_dB (int): maximum ripple amplitude in decibels

        If order and ripple_dB are not specified a value of 3 is used for both.

        """
        b, a = signal.cheby1(order, ripple_dB, [2.*np.pi*self.f0*1e9, 2.*np.pi*self.f1*1e9], 'bandpass', analog=True)
        w, h = signal.freqs(b, a, worN=self.freqs*2*np.pi*1e9)

        self.weights = abs(h)
        if normalize :
            A = self.get_normalization()
            self. weights  /=A
            self.isnormalized=True


    def get_normalization (self):
        """
        Estimate the integral over the frequency band
        """
        return np.trapz(self.weights,self.freqs )


    # Find effective central frequency of a bandpass profile
    def find_central_frequency(self):
        """Find the effective central frequency of
        a bandpass profile as defined in https://arxiv.org/abs/1303.5070
        """
        if self.isnormalized:
            return np.trapz(self.freqs*self.weights,self.freqs )
        else :
            return np.trapz (self.freqs*self.weights ,self.freqs)/self.get_normalization()

    @staticmethod
    def from_imo(imo: Imo, url: Union[UUID, str]):
        """Create a `BandPassInfo` object from a definition in the IMO
        The `url` must either specify a UUID or a full URL to the
        object.

        """
        obj = imo.query(url)
        return BandPassInfo.from_dict(obj.metadata)

    def interpolate_band  (self ):
        """
        This function aims at building the sampler in order to generate random samples
        statistically equivalent to the model bandpass
        """
        #normalize band

        if not  self.isnormalized:
            self.normalize_band()
        #Interpolate the band
        b = sp.interpolate.interp1d(x=self.freqs , y=self.weights )
        #estimate the CDF
        Pnu =np.array([sp.integrate.quad(b , a =self.freqs.min(), b=inu  )[0]  for inu in self.freqs[1:] ])
        #interpolate the inverse CDF
        self.Sampler = sp.interpolate.interp1d(Pnu ,self.freqs[:-1] + np.diff(self.freqs),
                                               bounds_error=False, fill_value="extrapolate")

    def bandpass_resampling(self,  bstrap_size= 1000, nresample=54 , model =None  ):
        """
        Resample a  bandpass with bootstrap resampling.
        Notice that the user can provide any sampler built with the `interpolate_band`
        method, if not provided an error will be raised!

        Args :
        - bstrap_size (int) : encodes the size of the random dataset  to be generated from the Sampler
        - nresample (int) : define how fine is the grid for the resampled bandpass
        """

        if model is not  None :
            print(f"Sampler  from {model.name }")
            Sampler= model.Sampler
        else:
            try :
                Sampler = self.Sampler
            except AttributeError :
                print("Can't resample if no sampler is built and/or provided, interpolating the band")
                self. interpolate_band()
                Sampler= self.Sampler



        X =  np.random.uniform(size=bstrap_size)
        bins_nu=np.linspace(self.freqs.min(), self.freqs.max(),nresample)
        h, xb =np.histogram(Sampler( X ), density=True ,bins= bins_nu   )

        nu_b  = xb[:-1] + np.diff(xb)
        resampled_bpass =abs(sp.interpolate.interp1d(nu_b, h, kind='cubic', bounds_error=False, fill_value="extrapolate")(self.freqs))
        if self.isnormalized:
            return   resampled_bpass/np.trapz(resampled_bpass,self.freqs )
        else:
            return resampled_bpass
