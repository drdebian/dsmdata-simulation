select
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.GridDraw
,a.runningtime
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PowerDemandTotal, 2)
	  else 0
	  end as SSR
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PVProducedTotal, 2)
	  else 0
	  end as SCR
from result1_direct b
inner join result0_direct a
       on a.label = b.label
	  and a.runningdate = b.runningdate
where b.[index] = 0
  and a.runningdate > (select max(timestamp) from lastrun)
  --and label like "direct_pv1.00_bess0.50%"
  
UNION

select
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.GridDraw
,a.runningtime
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PowerDemandTotal, 2)
	  else 0
	  end as SSR
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PVProducedTotal, 2)
	  else 0
	  end as SCR
from result1_rule b
inner join result0_rule a
       on a.label = b.label
	  and a.runningdate = b.runningdate
where b.[index] = 0
  and a.runningdate > (select max(timestamp) from lastrun)

UNION

select
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.GridDraw
,a.runningtime
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PowerDemandTotal, 2)
	  else 0
	  end as SSR
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PVProducedTotal, 2)
	  else 0
	  end as SCR
from result1_pred b
inner join result0_pred a
       on a.label = b.label
	  and a.runningdate = b.runningdate
where b.[index] = 0
  and a.runningdate > (select max(timestamp) from lastrun)

UNION

select
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.GridDraw
,a.runningtime
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PowerDemandTotal, 2)
	  else 0
	  end as SSR
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PVProducedTotal, 2)
	  else 0
	  end as SCR
from result2_stoch b
inner join result0_stoch a
       on a.label = b.label
	  and a.runningdate = b.runningdate
where b.period = 0
  and b.scenario = 0
  and a.runningdate > (select max(timestamp) from lastrun)

UNION

select
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.GridDraw
,a.runningtime
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PowerDemandTotal, 2)
	  else 0
	  end as SSR
,case when a.PVProducedTotal > 0
      then round((a.PowerDemandTotal - a.GridDrawTotal)/a.PVProducedTotal, 2)
	  else 0
	  end as SCR
from result1_perfect b
inner join result0_perfect a
       on a.label = b.label
	  and a.runningdate = b.runningdate
where b.[index] = 0
  and a.runningdate > (select max(timestamp) from lastrun)
