// The screenshot of the workbench in Fig 1A, taken with digitiser-workbench's own browser test:
//
//   mkdir -p WORKBENCH/_paper_tmp
//   cp validation/activity-traces/figures/PMC4001052-fig4.jpg WORKBENCH/_paper_tmp/lopatina2014-fig4.jpg
//   (cd WORKBENCH && python3 tests/browser_test.py PATH/TO/paper/workbench_shots.js 800 /tmp/wb)
//   cp /tmp/wb-marked.png paper/workbench-marked.png
//   rm -r WORKBENCH/_paper_tmp
//
// The page opens the figure, marks the two plot areas, the two trace colours and the ends of each x axis
// (0 and 168 h), and is photographed (marked).
await waitFor(() => MG.app.digitise, 20000, 'the Digitise tab');
const K = MG.kernels.get();
await waitFor(() => K.state === 'ready' || K.state === 'failed', 400000, 'Python');
log('kernel', K.state);
await waitFor(() => MG.app.digitise.warm, 180000, 'warm');
const D = MG.app.digitise;
window.D = D;
const blob = await (await fetch('/_paper_tmp/lopatina2014-fig4.jpg')).blob();
const file = new File([blob], 'lopatina2014-fig4.jpg', { type: 'image/jpeg' });
const paths = await MG.project.addFiles(MG.app.fs, MG.app.HOME + '/pictures', [file]);
log('added', JSON.stringify(paths));
await D.open(paths[0]);
await waitFor(() => D.found, 120000, 'describe');
D.apply({ kind: 'lines', marks: { areas: [[64.5, 36, 615.1, 253.1], [64.5, 346.8, 616, 563.3]], exclude: [], colours: ['#6f9bd4', '#d9a09f'],
  ticks: [{ axis: 'x', px: 64.5, value: 0, side: 'bottom', at: [64.5, 262] }, { axis: 'x', px: 615.1, value: 168, side: 'bottom', at: [615.1, 262] },
          { axis: 'x', px: 64.5, value: 0, side: 'bottom', at: [64.5, 572] }, { axis: 'x', px: 616, value: 168, side: 'bottom', at: [616, 572] }] } });
//--SHOT marked
return 'ok';
