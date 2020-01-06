import matplotlib as mpl


###############################################################################


small = 10
medium = 14
big = 16

# Give colours in hex
qubit     = '#6d78aa'  # qubit
cavity1   = '#c86464'  # cavity 1
cavity2   = '#c86464'  # cavity 2
coupling1 = '#68b799'  # coupling between qubit and cavity 1
coupling2 = '#7eb6a1'  # coupling between qubit and cavity 1
drive     = '#68b799'  # single-tone drive strength
driveq    = 'green'    # qubit-friendly drive tone strength
drivec    = 'cyan'     # cavity-friendly drive tone strenth
sbred     = '#c86464'  # P(e0)-P(g1), red sideband transitions
sbblue    = '#6196b3'  # P(e1)-P(g0), blue sideband transitions
hline     = 'gray'     # horizontal domain lines

# (line, marker) styles
larger  = ('-', 'x')
smaller = ('-.', '+')

alpha = 1  # opacity of lines

mpl.rcParams['font.size'] = medium
mpl.rcParams['axes.titlesize'] = medium
mpl.rcParams['axes.labelsize'] = medium
mpl.rcParams['xtick.labelsize'] = medium
mpl.rcParams['ytick.labelsize'] = medium
mpl.rcParams['legend.fontsize'] = small
mpl.rcParams['figure.titlesize'] = big
mpl.rcParams['lines.linewidth'] = 1.5
   

###############################################################################
 

plotcolours = {
    'qubit' : qubit,
    'cavity1' : cavity1,
    'cavity2' : cavity2,
    'cavity' : cavity1,
    'coupling1' : coupling1,
    'coupling2' : coupling2,
    'coupling' : coupling1,
    'drive' : drive,
    'driveq' : driveq,
    'drivec' : drivec,
    'sbred' : sbred,
    'sbblue' : sbblue,
    'hline' : hline
}

linestyles = {
    'larger' : larger[0],
    'smaller' : smaller[0]
}

markerstyles = {
    'larger' : larger[1],
    'smaller' : smaller[1]
}