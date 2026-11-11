/** Build the joint pose sliders UI and manage base pose offsets. */
export function createJointsPanel(joints, container = document.getElementById('sl')) {
  const pose = {};
  joints.forEach((j) => {
    pose[j[0]] = 0;
  });

  joints.forEach((j) => {
    const label = document.createElement('label');
    label.textContent = j[1];
    const input = document.createElement('input');
    input.type = 'range';
    input.min = j[4];
    input.max = j[5];
    input.step = 0.01;
    input.value = 0;
    input.oninput = () => {
      pose[j[0]] = +input.value;
    };
    input.dataset.k = j[0];
    label.appendChild(input);
    container.appendChild(label);
  });

  const resetPose = () => {
    joints.forEach((j) => {
      pose[j[0]] = 0;
    });
    container.querySelectorAll('input').forEach((input) => {
      input.value = 0;
    });
  };

  return { pose, resetPose };
}
