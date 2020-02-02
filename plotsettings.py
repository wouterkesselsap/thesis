import matplotlib as mpl


###############################################################################


figsize = (10, 5)

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
colormap  = 'gist_heat'   # colormap for color plots

# (line, marker) styles
larger       = ('-', 'x')
smaller      = ('-.', '+')
analytical   = ('-', 'x')
simulated    = ('--', '+')
experimental = ('-.', '.')

alpha = 1  # opacity of lines

mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.titlesize'] = 26
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['figure.titlesize'] = 26
mpl.rcParams['lines.linewidth'] = 3
   

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
    'hline' : hline,
    'colormap' : colormap
}

linestyles = {
    'larger' : larger[0],
    'smaller' : smaller[0],
    'analytical' : analytical[0],
    'simulated' : simulated[0],
    'experimental' : experimental[0]
}

markerstyles = {
    'larger' : larger[1],
    'smaller' : smaller[1],
    'analytical' : analytical[1],
    'simulated' : simulated[1],
    'experimental' : experimental[1]
}

mpl.rcParams['figure.figsize'] = figsize