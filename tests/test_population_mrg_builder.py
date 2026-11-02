"""Check the local transformation from pinned ModelDB HOC to population HOC."""

import unittest

from hpc_tools.build_population_mrg import population_template


class PopulationMRGBuilderTests(unittest.TestCase):
    def test_parameterizes_geometry_and_removes_gui_stimulus(self):
        upstream = '''load_proc("nrnmainmenu")
proc model_globels() {
    axonnodes=21
    paranodes1=40
    paranodes2=40
    axoninter=120
    axontotal=221
    fiberD=10.0 // choose diameter
}
model_globels ()
proc dependent_var() {
    if (fiberD==14.0) {g=0.739 deltax=1400}
}
dependent_var()
initialize()
node[10] stim = new IClamp(0.5)
xpanel("Stimulus parameters")
'''
        generated = population_template(upstream)
        self.assertIn('axonnodes=mrg_population_axonnodes', generated)
        self.assertIn('fiberD=mrg_population_fiberD', generated)
        self.assertIn('paranodes1=2*(axonnodes-1)', generated)
        self.assertIn('if (fiberD==14.0) {g=0.739 deltax=1400}', generated)
        self.assertNotIn('IClamp', generated)
        self.assertNotIn('xpanel', generated)

    def test_refuses_unknown_upstream_shape(self):
        with self.assertRaises(ValueError):
            population_template('proc model_globels() {}\n')


if __name__ == '__main__':
    unittest.main()
