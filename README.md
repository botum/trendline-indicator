# trendline-indicator
Ads trend-lines to the passed dataframe

# How?

This is first approach. Takes dataframe, process to peaks with a customized
zig-zag library implementation and then finds trends with those points.

Trends are find through a simple loop over the highs and lows, called 'A'
(ax,ay), and the peaks that follow them called 'B' (bx, by) that are "seen" from
the A point. I call this the Light House Trend Line, there might be a term
already for this sure, but this is all about imagination, mainly spatial.

Then we'll be adding more parameters to the processing to have finest trends
needed for strategies.

# To do

* Paramters:
  * tolerance: Tolerance for difference tolerable for points
  * confirmations:  how many low/high where exactly in the sup/res trend respectively.
  * slope_min: minimal angle of valid trendlines
  * slope_max: maximum angle of valid trendlines
  * Log scale option
