(function() {

  // Chart design based on the recommendations of Stephen Few. Implementation
  d3.bullet = function() {

    var orient = "left", // TODO top & bottom
        reverse = false,
        ticks = bulletTicks,
        markers = bulletMarkers,
        startPoints = bulletStartPoints,
        colors = bulletColors,
        widths = bulletWidths,
        width = 380,
        height = 30,
        tickFormat = null;
              
    // For each small bullet chart
    function bullet(g) {
      
      g.each(function(d, i) {

        var tickValue = ticks.call(this, d, i).slice().sort(d3.descending),
            markerValue = markers.call(this, d, i).slice().sort(d3.descending),
            g = d3.select(this);

        // Compute the new x-scale.
        var x1 = d3.scale.linear()
            .domain([tickValue[tickValue.length-1], tickValue[0]])
            .range(reverse ? [width, 0] : [0, width]);


        // Retrieve the old x-scale, if this is an update.
        var x0 = this.__chart__ || d3.scale.linear()
            .domain([tickValue[tickValue.length-1], Infinity])
            .range(x1.range());

            // Stash the new scale.
        this.__chart__ = x1;

        // Update the measure rects.
        var measure = g.selectAll("rect.measure");
        measure.data(startPoints).enter().append("rect") // load measure data
            .attr("class", function(d, i) { return "measure s" + i; })
            .attr("height", height / 3)
            .attr("x", function(d) {return d; })
            .attr("y", height / 3)
            .data(widths) // load widths data
            .attr("width", function(d) {return d; }) //
            .data(colors) // load color data [color1, color2, color3 ...]
            .attr("fill", function(d) {return d; }) // fill the color
            .data(startPoints) // load measure data
            .attr("x", function(d) {return d; })
            .data(widths) // load widths data
            .attr("width", function(d) {return d; });

        // Update the marker lines.
        var marker = g.selectAll("line.marker")
            .data(markerValue);

        marker.enter().append("line")
            .attr("class", "marker")
            .attr("x1", x0)
            .attr("x2", x0)
            .attr("y1", height / 6)
            .attr("y2", height * 5 / 6)
            .attr("x1", x1)
            .attr("x2", x1)
            .datum({x: 0, y: 0});

        // Compute the tick format.
        var format = tickFormat || x1.tickFormat(8);

        // Update the tick groups.
        var tick = g.selectAll("g.tick")
            .data(x1.ticks(8), function(d) {
              return this.textContent || format(d);
            });
        // Initialize the ticks with the old scale, x0.
        var tickEnter = tick.enter().append("g")
            .attr("class", "tick")
            .attr("transform", bulletTranslate(x0))
            .style("opacity", 1e-6);
  
        tickEnter.append("line")
            .attr("y1", height)
            .attr("y2", height * 7 / 6);
  
        tickEnter.append("text")
            .attr("text-anchor", "middle")
            .attr("dy", "1em")
            .attr("y", height * 7 / 6)
            .text(format);

        // Transition the entering ticks to the new scale, x1.
        tickEnter.attr("transform", bulletTranslate(x1))
            .style("opacity", 1);
  
        // Transition the updating ticks to the new scale, x1.
        var tickUpdate = tick.attr("transform", bulletTranslate(x1))
            .style("opacity", 1);
  
        tickUpdate.select("line")
            .attr("y1", height)
            .attr("y2", height * 7 / 6);
  
        tickUpdate.select("text")
            .attr("y", height * 7 / 6);
      });
    }
  
    // left, right, top, bottom
    bullet.orient = function(x) {
      if (!arguments.length) return orient;
      orient = x;
      reverse = orient == "right" || orient == "bottom";
      return bullet;
    };
  
    // ticks (bad, satisfactory, good)
    bullet.ticks = function(x) {
      if (!arguments.length) return ticks;
      ticks = x;
      return bullet;
    };
  
    // markers (previous, goal)
    bullet.markers = function(x) {
      if (!arguments.length) return markers;
      markers = x;
      return bullet;
    };
  
    // measures (actual, forecast)
    bullet.startPoints = function(x) {
      if (!arguments.length) return startPoints;
      startPoints = x;
      return bullet;
    };
  
    bullet.width = function(x) {
      if (!arguments.length) return width;
      width = x;
      return bullet;
    };
  
    bullet.height = function(x) {
      if (!arguments.length) return height;
      height = x;
      return bullet;
    };
  
    bullet.tickFormat = function(x) {
      if (!arguments.length) return tickFormat;
      tickFormat = x;
      return bullet;
    };
  
    return bullet;
  };
  
  function bulletTicks(d) {
    return d.ticks;
  }
  
  function bulletMarkers(d) {
    return d.markers;
  }
  
  function bulletStartPoints(d) {
    return d.startPoints;
  }

  function bulletColors(d) {
    return d.colors;
  }

  function bulletWidths(d) {
    return d.widths;
  }

  function bulletTranslate(x) {
    return function(d) {
      return "translate(" + x(d) + ",0)";
    };
  }
  })();