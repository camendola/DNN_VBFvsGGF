
#tautau BDT
#python training/VBFvsGGF.py --cv_BDT --show --selection "type"
#python training/VBFvsGGF.py --cv_BDT --show --selection "type-baseline"
#python training/VBFvsGGF.py --cv_BDT --show --selection  "type-baseline-looseVBF" --channel 2
python training/VBFvsGGF.py --cv_BDT --show --selection  "type-baseline-looseVBF" --channel 2


#tautau DNN
#python training/VBFvsGGF.py --optimize --selection "type" 
#python training/VBFvsGGF.py --optimize --selection "type-baseline" 
#python training/VBFvsGGF.py --optimize --selection "type-baseline-looseVBF"
#python training/VBFvsGGF.py --optimize --selection "type-baseline-tightVBF"


#python training/VBFvsGGF.py --optimize --selection "type" 
#python training/VBFvsGGF.py --optimize --channel 1 --selection "type-baseline" 
#python training/VBFvsGGF.py --optimize --channel 1 --selection "type-baseline-looseVBF"

 
#python training/VBFvsGGF.py --optimize_BDT --selection "type" 
#python training/VBFvsGGF.py --optimize_BDT --selection "type-baseline"
#python training/VBFvsGGF.py --optimize_BDT --selection "type-baseline-tightVBF"

#python training/VBFvsGGF.py --optimize_BDT --channel 1 --selection "type" 
#python training/VBFvsGGF.py --optimize_BDT --channel 1 --selection "type-baseline" 
#python training/VBFvsGGF.py --optimize_BDT --channel 1 --selection "type-baseline-looseVBF"


#python training/VBFvsGGF.py --cv --selection  "type-baseline-looseVBF" --channel 0
#python training/VBFvsGGF.py --optimize --selection  "type-baseline-looseVBF" --channel 1
#python training/VBFvsGGF.py --cv --show --selection  "type-baseline-looseVBF" --channel 2
#python training/VBFvsGGF.py --optimize --show --selection  "type-baseline-tightVBF" --channel 2
