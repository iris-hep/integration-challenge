# p7018 is newest ongoing bulk production, p6697 is previous bulk production
# see https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/DerivationProductionTeam#p_tags

containers = {
    #
    # data
    #
    "data": [
        "data15_13TeV:data15_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp15_v01_p6697",
        "data16_13TeV:data16_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp16_v01_p6697",
        "data17_13TeV:data17_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp17_v01_p6697",
        "data18_13TeV:data18_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp18_v01_p6697",
        "data22_13p6TeV:data22_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp22_v02_p6700",
        "data23_13p6TeV:data23_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp23_v01_p6700",
        "data24_13p6TeV:data24_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp24_v01_p6700",
    ],

    #
    # ttbar baseline, allhad removed, should include var3c variation + muR/muF as weights
    #
    "ttbar_nom": [
        # centralpage --scope=mc20_13TeV --physlite Top TTbar Baseline PowhegPythia
        "mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13167_r13146_p6697",
        "mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13144_r13146_p6697",
        "mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbar Baseline PowhegPythia
        "mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.601230.PhPy8EG_A14_ttbar_hdamp258p75_dil.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601230.PhPy8EG_A14_ttbar_hdamp258p75_dil.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601230.PhPy8EG_A14_ttbar_hdamp258p75_dil.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],

    #
    # ttbar systematics
    #
    "ttbar_H7": [
        # centralpage --scope=mc20_13TeV --physlite Top TTbar Alternative PowhegHerwig713
        "mc20_13TeV.411233.PowhegHerwig7EvtGen_tt_hdamp258p75_713_SingleLep.deriv.DAOD_PHYSLITE.e7580_a907_r14859_p6697",
        "mc20_13TeV.411233.PowhegHerwig7EvtGen_tt_hdamp258p75_713_SingleLep.deriv.DAOD_PHYSLITE.e7580_a907_r14860_p6697",
        "mc20_13TeV.411233.PowhegHerwig7EvtGen_tt_hdamp258p75_713_SingleLep.deriv.DAOD_PHYSLITE.e7580_a907_r14861_p6697",
        "mc20_13TeV.411234.PowhegHerwig7EvtGen_tt_hdamp258p75_713_dil.deriv.DAOD_PHYSLITE.e7580_a907_r14859_p6697",
        "mc20_13TeV.411234.PowhegHerwig7EvtGen_tt_hdamp258p75_713_dil.deriv.DAOD_PHYSLITE.e7580_a907_r14860_p6697",
        "mc20_13TeV.411234.PowhegHerwig7EvtGen_tt_hdamp258p75_713_dil.deriv.DAOD_PHYSLITE.e7580_a907_r14861_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbar Systematic PowhegHerwig7
        "mc23_13p6TeV.601414.PhH7EG_A14_ttbar_hdamp258p75_Single_Lep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601414.PhH7EG_A14_ttbar_hdamp258p75_Single_Lep.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601414.PhH7EG_A14_ttbar_hdamp258p75_Single_Lep.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.601415.PhH7EG_A14_ttbar_hdamp258p75_Dilep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601415.PhH7EG_A14_ttbar_hdamp258p75_Dilep.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601415.PhH7EG_A14_ttbar_hdamp258p75_Dilep.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],
    "ttbar_hdamp": [
        # centralpage --scope=mc20_13TeV --physlite Top TTbar Systematic Hdamp
        "mc20_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e6454_a907_r14859_p6697",
        "mc20_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e6454_a907_r14860_p6697",
        "mc20_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e6454_a907_r14861_p6697",
        "mc20_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e6454_a907_r14859_p6697",
        "mc20_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e6454_a907_r14860_p6697",
        "mc20_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e6454_a907_r14861_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbar Systematic PowhegPythia8_hdamp
        "mc23_13p6TeV.601398.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601398.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601398.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.601399.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.601399.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.601399.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],
    "ttbar_pthard": [
        # TODO (low priority) incomplete in mc23
        # centralpage --scope=mc20_13TeV --physlite Top TTbar Systematic pThard1
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbar Systematic PowhegPythia8_pTHard1
    ],

    #
    # single top baseline
    #
    "st_schan": [
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Baseline PowhegPythia-schan
        "mc20_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_PHYSLITE.e6527_s3681_r13167_r13146_p6697",
        "mc20_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_PHYSLITE.e6527_s3681_r13144_r13146_p6697",
        "mc20_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_PHYSLITE.e6527_s3681_r13145_r13146_p6697",
        "mc20_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_PHYSLITE.e6527_s3681_r13167_r13146_p6697",
        "mc20_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_PHYSLITE.e6527_s3681_r13144_r13146_p6697",
        "mc20_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_PHYSLITE.e6527_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Baseline PowhegPythia8-schan
        "mc23_13p6TeV.601348.PhPy8EG_tb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4162_r14622_p6697",
        "mc23_13p6TeV.601348.PhPy8EG_tb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.601348.PhPy8EG_tb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.601349.PhPy8EG_tb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4162_r14622_p6697",
        "mc23_13p6TeV.601349.PhPy8EG_tb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.601349.PhPy8EG_tb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],
    "st_tchan": [
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Baseline PowhegPythia-tchan
        # hadronic top decays removed
        "mc20_13TeV.410658.PhPy8EG_A14_tchan_BW50_lept_top.deriv.DAOD_PHYSLITE.e6671_s3681_r13167_r13146_p6697",
        "mc20_13TeV.410658.PhPy8EG_A14_tchan_BW50_lept_top.deriv.DAOD_PHYSLITE.e6671_s3681_r13144_r13146_p6697",
        "mc20_13TeV.410658.PhPy8EG_A14_tchan_BW50_lept_top.deriv.DAOD_PHYSLITE.e6671_s3681_r13145_r13146_p6697",
        "mc20_13TeV.410659.PhPy8EG_A14_tchan_BW50_lept_antitop.deriv.DAOD_PHYSLITE.e6671_s3681_r13167_r13146_p6697",
        "mc20_13TeV.410659.PhPy8EG_A14_tchan_BW50_lept_antitop.deriv.DAOD_PHYSLITE.e6671_s3681_r13144_r13146_p6697",
        "mc20_13TeV.410659.PhPy8EG_A14_tchan_BW50_lept_antitop.deriv.DAOD_PHYSLITE.e6671_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Baseline PowhegPythia8-tchan
        "mc23_13p6TeV.601350.PhPy8EG_tqb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4162_r14622_p6697",
        "mc23_13p6TeV.601350.PhPy8EG_tqb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.601350.PhPy8EG_tqb_lep_antitop.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.601351.PhPy8EG_tqb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4162_r14622_p6697",
        "mc23_13p6TeV.601351.PhPy8EG_tqb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.601351.PhPy8EG_tqb_lep_top.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],
    "Wt": [
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Baseline PowhegPythia8-DynScale-Wt
        "mc20_13TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_s4231_r13167_p6697",
        "mc20_13TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_s4231_r13144_p6697",
        "mc20_13TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_s4231_r13145_p6697",
        "mc20_13TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_s4231_r13167_p6697",
        "mc20_13TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_s4231_r13144_p6697",
        "mc20_13TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_s4231_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Baseline PowhegPythia8-tW
        "mc23_13p6TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_s4162_r14622_p6697",
        "mc23_13p6TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_s4159_r15224_p6697",
        "mc23_13p6TeV.601352.PhPy8EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_s4369_r16083_p6697",
        "mc23_13p6TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_s4162_r14622_p6697",
        "mc23_13p6TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_s4159_r15224_p6697",
        "mc23_13p6TeV.601355.PhPy8EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_s4369_r16083_p6697",
    ],

    #
    # single top systematics
    #
    "Wt_DS": [
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Systematic PowhegPythia8-DynScale-DS-Wt
        "mc20_13TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8482_s3681_r13167_r13146_p6697",
        "mc20_13TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8482_s3681_r13144_r13146_p6697",
        "mc20_13TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8482_s3681_r13145_r13146_p6697",
        "mc20_13TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8482_s3681_r13167_r13146_p6697",
        "mc20_13TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8482_s3681_r13144_r13146_p6697",
        "mc20_13TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8482_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Systematic PowhegPythia8-tW_DS
        "mc23_13p6TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8549_s4162_r15540_p6697",
        "mc23_13p6TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8549_s4159_r15530_p6697",
        "mc23_13p6TeV.601627.PhPy8EG_tW_DS_dyn_incl_antitop.deriv.DAOD_PHYSLITE.e8549_s4369_r16083_p6697",
        "mc23_13p6TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8549_s4162_r15540_p6697",
        "mc23_13p6TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8549_s4159_r15530_p6697",
        "mc23_13p6TeV.601631.PhPy8EG_tW_DS_dyn_incl_top.deriv.DAOD_PHYSLITE.e8549_s4369_r16083_p6697",
    ],
    "Wt_H7": [
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Systematic PowhegHerwig7-DynScale-Wt
        "mc20_13TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_a907_r14859_p6697",
        "mc20_13TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_a907_r14860_p6697",
        "mc20_13TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8547_a907_r14861_p6697",
        "mc20_13TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_a907_r14859_p6697",
        "mc20_13TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_a907_r14860_p6697",
        "mc20_13TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8547_a907_r14861_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Systematic PowhegHerwig7-tW
        "mc23_13p6TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_a910_r15540_p6697",
        "mc23_13p6TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_a911_r15530_p6697",
        "mc23_13p6TeV.602235.PhH7EG_tW_dyn_DR_incl_top.deriv.DAOD_PHYSLITE.e8551_a934_r16083_p6697",
        "mc23_13p6TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_a910_r15540_p6697",
        "mc23_13p6TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_a911_r15530_p6697",
        "mc23_13p6TeV.602236.PhH7EG_tW_dyn_DR_incl_antitop.deriv.DAOD_PHYSLITE.e8551_a934_r16083_p6697",
    ],
    "Wt_pthard": [
        # TODO (low priority) incomplete in mc23
        # centralpage --scope=mc20_13TeV --physlite Top SingleTop Systematic pthard1-DynScale-Wt
        # centralpage --scope=mc23_13p6TeV --physlite Top SingleTop Systematic PowhegPythia8-tW_pTHard1
    ],

    #
    # ttV
    #
    "ttV": [
        # centralpage --scope=mc20_13TeV --physlite Top TTbarX Baseline MC20-Sherpa2214-ttW
        "mc20_13TeV.701260.Sh_ttW_0LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13167_p6697",
        "mc20_13TeV.701260.Sh_ttW_0LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13144_p6697",
        "mc20_13TeV.701260.Sh_ttW_0LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13145_p6697",
        "mc20_13TeV.701261.Sh_ttW_1LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13167_p6697",
        "mc20_13TeV.701261.Sh_ttW_1LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13144_p6697",
        "mc20_13TeV.701261.Sh_ttW_1LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13145_p6697",
        "mc20_13TeV.701262.Sh_ttW_2LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13167_p6697",
        "mc20_13TeV.701262.Sh_ttW_2LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13144_p6697",
        "mc20_13TeV.701262.Sh_ttW_2LxAODFilt.deriv.DAOD_PHYSLITE.e8577_s3797_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbarX Baseline Sherpa2214-ttW
        "mc23_13p6TeV.700995.Sh_2214_ttW_0Lfilter.deriv.DAOD_PHYSLITE.e8532_s4162_r14622_p6697",
        "mc23_13p6TeV.700995.Sh_2214_ttW_0Lfilter.deriv.DAOD_PHYSLITE.e8532_s4159_r15224_p6697",
        "mc23_13p6TeV.700995.Sh_2214_ttW_0Lfilter.deriv.DAOD_PHYSLITE.e8532_s4369_r16083_p6697",
        "mc23_13p6TeV.700996.Sh_2214_ttW_1Lfilter.deriv.DAOD_PHYSLITE.e8532_s4162_r14622_p6697",
        "mc23_13p6TeV.700996.Sh_2214_ttW_1Lfilter.deriv.DAOD_PHYSLITE.e8532_s4159_r15224_p6697",
        "mc23_13p6TeV.700996.Sh_2214_ttW_1Lfilter.deriv.DAOD_PHYSLITE.e8532_s4369_r16083_p6697",
        "mc23_13p6TeV.700997.Sh_2214_ttW_2Lfilter.deriv.DAOD_PHYSLITE.e8532_s4162_r14622_p6697",
        "mc23_13p6TeV.700997.Sh_2214_ttW_2Lfilter.deriv.DAOD_PHYSLITE.e8532_s4159_r15224_p6697",
        "mc23_13p6TeV.700997.Sh_2214_ttW_2Lfilter.deriv.DAOD_PHYSLITE.e8532_s4369_r16083_p6697",
        # centralpage --scope=mc20_13TeV --physlite Top TTbarX Baseline aMcAtNloPythia-ttZ
        "mc20_13TeV.504338.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq.deriv.DAOD_PHYSLITE.e8255_s3797_r13167_p6697",
        "mc20_13TeV.504338.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq.deriv.DAOD_PHYSLITE.e8255_s3797_r13144_p6697",
        "mc20_13TeV.504338.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq.deriv.DAOD_PHYSLITE.e8255_s3797_r13145_p6697",
        "mc20_13TeV.504346.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu.deriv.DAOD_PHYSLITE.e8255_s3797_r13167_p6697",
        "mc20_13TeV.504346.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu.deriv.DAOD_PHYSLITE.e8255_s3797_r13144_p6697",
        "mc20_13TeV.504346.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu.deriv.DAOD_PHYSLITE.e8255_s3797_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbarX Baseline aMcAtNloPythia-ttZ
        "mc23_13p6TeV.522036.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq_run3.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.522036.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq_run3.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.522036.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZqq_run3.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
        "mc23_13p6TeV.522040.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu_run3.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.522040.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu_run3.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.522040.aMCPy8EG_NNPDF30NLO_A14N23LO_ttZnunu_run3.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
        # centralpage --scope=mc20_13TeV --physlite Top TTbarX Baseline aMcAtNloPythia-ttll
        "mc20_13TeV.504330.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee.deriv.DAOD_PHYSLITE.e8255_s3797_r13167_p6697",
        "mc20_13TeV.504330.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee.deriv.DAOD_PHYSLITE.e8255_s3797_r13144_p6697",
        "mc20_13TeV.504330.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee.deriv.DAOD_PHYSLITE.e8255_s3797_r13145_p6697",
        "mc20_13TeV.504334.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu.deriv.DAOD_PHYSLITE.e8255_s3797_r13167_p6697",
        "mc20_13TeV.504334.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu.deriv.DAOD_PHYSLITE.e8255_s3797_r13144_p6697",
        "mc20_13TeV.504334.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu.deriv.DAOD_PHYSLITE.e8255_s3797_r13145_p6697",
        "mc20_13TeV.504342.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau.deriv.DAOD_PHYSLITE.e8255_s3797_r13167_p6697",
        "mc20_13TeV.504342.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau.deriv.DAOD_PHYSLITE.e8255_s3797_r13144_p6697",
        "mc20_13TeV.504342.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau.deriv.DAOD_PHYSLITE.e8255_s3797_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top TTbarX Baseline aMcAtNloPythia-ttll
        "mc23_13p6TeV.522024.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee_run3.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.522024.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee_run3.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.522024.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee_run3.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
        "mc23_13p6TeV.522028.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu_run3.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.522028.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu_run3.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.522028.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu_run3.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
        "mc23_13p6TeV.522032.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau_run3.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.522032.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau_run3.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.522032.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau_run3.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
    ],

    #
    # rare / other top: tZ, tWZ, 4top
    #
    "rare_top": [
        # centralpage --scope=mc20_13TeV --physlite Top RareTop Baseline aMcAtNloPythia-tllq
        "mc20_13TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13167_p6697",
        "mc20_13TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13144_p6697",
        "mc20_13TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13145_p6697",
        "mc20_13TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13167_p6697",
        "mc20_13TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13144_p6697",
        "mc20_13TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s3797_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top RareTop Baseline aMcAtNloPythia-tllq
        "mc23_13p6TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4162_r15540_p6697",
        "mc23_13p6TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4159_r15530_p6697",
        "mc23_13p6TeV.545027.aMCPy8EG_tllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4369_r16083_p6697",
        "mc23_13p6TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4162_r15540_p6697",
        "mc23_13p6TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4159_r15530_p6697",
        "mc23_13p6TeV.545028.aMCPy8EG_tbarllq_4FS_HT6.deriv.DAOD_PHYSLITE.e8564_s4369_r16083_p6697",
        # centralpage --scope=mc20_13TeV --physlite Top RareTop Baseline aMcAtNloPythia-tWZ
        "mc20_13TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8553_s4231_r13167_r13146_p6697",
        "mc20_13TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8553_s4231_r13144_r13146_p6697",
        "mc20_13TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8553_s4231_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite Top RareTop Baseline aMcAtNloPythia-tWZ
        "mc23_13p6TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8558_s4162_r14622_p6697",
        "mc23_13p6TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8558_s4159_r15224_p6697",
        "mc23_13p6TeV.525955.aMCPy8EG_tWZ_Ztoll_DR1.deriv.DAOD_PHYSLITE.e8558_s4369_r16083_p6697",
        # TODO (low priority) 4top incomplete in m23e and NLO vs LO
        # centralpage --scope=mc20_13TeV --physlite Top RareTop Baseline MadGraphPythia-tttt
        # centralpage --scope=mc23_13p6TeV --physlite Top RareTop Baseline aMcAtNloPythia-tttt
    ],

    #
    # W+jets
    #
    "wjets": [
        # centralpage --scope=mc20_13TeV --physlite WeakBoson Wjets_lv Baseline Sherpa_2211_W
        # hadronic tau decays removed
        "mc20_13TeV.700338.Sh_2211_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700338.Sh_2211_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700338.Sh_2211_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700339.Sh_2211_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700339.Sh_2211_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700339.Sh_2211_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700340.Sh_2211_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700340.Sh_2211_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700340.Sh_2211_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700341.Sh_2211_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700341.Sh_2211_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700341.Sh_2211_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700342.Sh_2211_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700342.Sh_2211_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700342.Sh_2211_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700343.Sh_2211_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700343.Sh_2211_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700343.Sh_2211_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700344.Sh_2211_Wtaunu_L_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700344.Sh_2211_Wtaunu_L_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700344.Sh_2211_Wtaunu_L_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700345.Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700345.Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700345.Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700346.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700346.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700346.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite WeakBoson Vjets Baseline Sherpa_2214
        # this includes both Z+jets and W+jets
        # dropping mW_105_ECMS samples
        "mc23_13p6TeV.700777.Sh_2214_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700777.Sh_2214_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700777.Sh_2214_Wenu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700778.Sh_2214_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700778.Sh_2214_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700778.Sh_2214_Wenu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700779.Sh_2214_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700779.Sh_2214_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700779.Sh_2214_Wenu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700780.Sh_2214_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700780.Sh_2214_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700780.Sh_2214_Wmunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700781.Sh_2214_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700781.Sh_2214_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700781.Sh_2214_Wmunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700782.Sh_2214_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700782.Sh_2214_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700782.Sh_2214_Wmunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700783.Sh_2214_Wtaunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700783.Sh_2214_Wtaunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700783.Sh_2214_Wtaunu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700784.Sh_2214_Wtaunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700784.Sh_2214_Wtaunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700784.Sh_2214_Wtaunu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700785.Sh_2214_Wtaunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700785.Sh_2214_Wtaunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700785.Sh_2214_Wtaunu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],

    #
    # Z+jets
    #
    "zjets": [
        # centralpage --scope=mc20_13TeV --physlite WeakBoson Zjets_ll Baseline Sherpa_2211_Z
        # Z>nunu removed
        "mc20_13TeV.700320.Sh_2211_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700320.Sh_2211_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700320.Sh_2211_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700321.Sh_2211_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700321.Sh_2211_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700321.Sh_2211_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700322.Sh_2211_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700322.Sh_2211_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700322.Sh_2211_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700323.Sh_2211_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700323.Sh_2211_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700323.Sh_2211_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        "mc20_13TeV.700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13167_r13146_p6697",
        "mc20_13TeV.700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_r13146_p6697",
        "mc20_13TeV.700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc20_13TeV --physlite WeakBoson Zjets_ll Baseline Sherpa_2214_Z
        "mc20_13TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8527_s3797_r13167_p7018",
        "mc20_13TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8527_s3797_r13144_p6697",
        "mc20_13TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8527_s3797_r13145_p6697",
        "mc20_13TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8527_s3797_r13167_p7018",
        "mc20_13TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8527_s3797_r13144_p7018",
        "mc20_13TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8527_s3797_r13145_p6697",
        "mc20_13TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8527_s3797_r13167_p7018",
        "mc20_13TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8527_s3797_r13144_p7018",
        "mc20_13TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8527_s3681_r13145_r13146_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite WeakBoson Vjets Baseline Sherpa_2214
        # this includes both Z+jets and W+jets
        # dropping mZ_105_ECMS and Mll10_40 samples, as well as Z>nunu
        "mc23_13p6TeV.700786.Sh_2214_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700786.Sh_2214_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700786.Sh_2214_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700787.Sh_2214_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700787.Sh_2214_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700787.Sh_2214_Zee_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700788.Sh_2214_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700788.Sh_2214_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700788.Sh_2214_Zee_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700789.Sh_2214_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6491",
        "mc23_13p6TeV.700789.Sh_2214_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700789.Sh_2214_Zmumu_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700790.Sh_2214_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6491",
        "mc23_13p6TeV.700790.Sh_2214_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700790.Sh_2214_Zmumu_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700791.Sh_2214_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700791.Sh_2214_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700791.Sh_2214_Zmumu_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15224_p6697",
        "mc23_13p6TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
        "mc23_13p6TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697",
        "mc23_13p6TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4159_r15530_p6697",
        "mc23_13p6TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8514_s4369_r16083_p6697",
    ],

    #
    # diboson (QCD/EW split in mc20 only)
    #
    "diboson": [
        # centralpage --scope=mc20_13TeV --physlite WeakBoson Diboson Baseline Sherpa_2214_VV_QCD
        "mc20_13TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        # centralpage --scope=mc20_13TeV --physlite WeakBoson Diboson Baseline Sherpa_2214_VV_EWK
        "mc20_13TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        "mc20_13TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13167_p6697",
        "mc20_13TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13144_p6697",
        "mc20_13TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8547_s3797_r13145_p6697",
        # centralpage --scope=mc23_13p6TeV --physlite WeakBoson Diboson Baseline Sherpa_2214
        "mc23_13p6TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701000.Sh_2214_lllljj.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701005.Sh_2214_lllvjj.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701010.Sh_2214_llvvjj_os.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701015.Sh_2214_llvvjj_ss.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701020.Sh_2214_lllljj_Int.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701025.Sh_2214_lllvjj_Int.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701030.Sh_2214_llvvjj_os_Int.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701035.Sh_2214_llvvjj_ss_Int.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701055.Sh_2214_llvv_ss.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701085.Sh_2214_ZqqZll.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701090.Sh_2214_ZbbZll.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701095.Sh_2214_ZqqZvv.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701100.Sh_2214_ZbbZvv.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701105.Sh_2214_WqqZll.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701110.Sh_2214_WqqZvv.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701115.Sh_2214_WlvZqq.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701120.Sh_2214_WlvZbb.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
        "mc23_13p6TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8543_s4162_r14622_p6697",
        "mc23_13p6TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8543_s4159_r15224_p6697",
        "mc23_13p6TeV.701125.Sh_2214_WlvWqq.deriv.DAOD_PHYSLITE.e8543_s4369_r16083_p6697",
    ]
}
