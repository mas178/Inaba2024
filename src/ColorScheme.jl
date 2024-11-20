module ColorScheme

using Colors
using Plots

const DARK_BLUE = colorant"#2D579A"
const DARK_RED = colorant"#B32034"

const RED = "#C7243A"
const ORANGE = "#EDAD0B"
const GREEN = "#23AC0E"
const BLUE = "#3261AB"
const PURPLE = "#744199"
const LIGHT_RED = "#DBBEC1"
const LIME = "#D8E212"
const LIGHT_GREE = "#81D674"
const LIGHT_BLUE = "#9BADCB"
const GRAY = "#C6C6C6"
const BLACK = "#000000"

const COLOR_GRAD = cgrad([DARK_RED, GRAY, DARK_BLUE], [0.0, 0.5, 1.0])
const COLORS = [RED, ORANGE, GREEN, BLUE, PURPLE, LIGHT_RED, LIME, LIGHT_GREE, LIGHT_BLUE, GRAY, BLACK]

end # end of module