import unittest
import numpy as np
import sop

class OptPricingTest(unittest.TestCase):

    def setUp(self):
    	self.K = 100
    	self.S = 100
    	self.rf = 0.02
    	self.T = 30.0 / 365.0
    	self.sigma = 0.35

    def test_bsm(self):
        #S, K = 100, 100
        #rf = 0.02
        #T = 30.0/365 # 
        #sigma = 0.35
        c = sop.BSM(self.S, self.K, self.T, self.rf, self.sigma, 'C') # B-S Call
        self.assertAlmostEqual(c, 4.08077, 5)
        p = sop.BSM(self.S, self.K, self.T, self.rf, self.sigma, 'P') # B-S Put
        self.assertAlmostEqual(p, 3.916522, 5)
        self.assertAlmostEqual(c - p, self.S - self.K*np.exp(-self.rf*self.T), 6)
        
    def test_eurobin(self):
        c = sop.EuroBin(self.S, self.K, self.T, self.rf, self.sigma, 200, 'C')
        p = sop.EuroBin(self.S, self.K, self.T, self.rf, self.sigma, 200, 'P')
        self.assertAlmostEqual(c, 4.08, delta=0.05)
        self.assertAlmostEqual(p, 3.92, delta=0.05)


if __name__ == "__main__":
	unittest.main()
