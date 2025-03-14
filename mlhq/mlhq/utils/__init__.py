import sys 


def proceed():                                                                     
    _proceed = input("\nContinue (y/n)? ")                                         
    if _proceed in ["Y", 'y']: return                                              
    print("Exiting")                                                               
    sys.exit(0)      
