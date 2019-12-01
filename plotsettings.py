import matplotlib as mpl


###############################################################################


small = 14
medium = 16
big = 18

# Give colours in hex
qubit     = '#868a9e'  # qubit
cavity1   = '#d29b9b'  # cavity 1
cavity2   = '#8ca8b8'  # cavity 2
coupling1 = '#7eb6a1'  # coupling between qubit and cavity 1
coupling2 = '#7eb6a1'  # coupling between qubit and cavity 1
drive     = '#7eb6a1'  # single-tone drive strength
driveq    = 'green'    # qubit-friendly drive tone strength
drivec    = 'cyan'     # cavity-friendly drive tone strenth
sbred     = '#d29b9b'  # P(e0)-P(g1), red sideband transitions
sbblue    = '#8ca8b8'  # P(e1)-P(g0), blue sideband transitions
hline     = 'gray'     # horizontal domain lines

# Other plot settings
alpha = 1  # opacity of lines


###############################################################################


mpl.rcParams['font.size'] = medium
mpl.rcParams['axes.titlesize'] = big
mpl.rcParams['axes.labelsize'] = medium
mpl.rcParams['xtick.labelsize'] = medium
mpl.rcParams['ytick.labelsize'] = medium
mpl.rcParams['legend.fontsize'] = medium
mpl.rcParams['figure.titlesize'] = big
mpl.rcParams['lines.linewidth'] = 3
    
plotcolours = {
    'qubit' : qubit,
    'cavity1' : cavity1,
    'cavity2' : cavity2,
    'coupling1' : coupling1,
    'coupling2' : coupling2,
    'drive' : drive,
    'driveq' : driveq,
    'drivec' : drivec,
    'sbred' : sbred,
    'sbblue' : sbblue,
    'hline' : hline
}